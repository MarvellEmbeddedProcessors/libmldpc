/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2022 Marvell.
 */

#include <getopt.h>
#include <linux/limits.h>
#include <stdio.h>

#include <jansson.h>

#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <rte_mldev.h>

#include <tvm_dpdk_mldev.h>

/* constants */
#define ML_DEFAULT_NUM_THREADS 1
#define ML_MAX_EAL_ARGS	       64
#define ML_OP_POOL_SIZE	       1024
#define ML_MAX_DESC_PER_QP     1

/* log stream */
#define RTE_LOGTYPE_MLDEV RTE_LOGTYPE_USER1

/* default EAL config */
#define LIBMLDPC_CONFIG_DEFAULT_PATH "/usr/share/libmldpc/config.json"

/* device context */
typedef struct ml_dev_ctx {
	int16_t dev_id;
	struct rte_ml_dev_info dev_info;
	struct rte_ml_dev_config dev_config;
	struct rte_mempool *op_pool;
} ml_dev_ctx_t;

ml_dev_ctx_t dev_ctx;

/* EAL variables */
int eal_argc;
char eal_args[ML_MAX_EAL_ARGS][PATH_MAX];
char **eal_argv;

/* Maximum number of layers per model */
#define ML_MAX_LAYERS 32

/* ML model variables structure */
typedef struct {
	struct rte_ml_buff_seg *i_q_segs[32];
	struct rte_ml_buff_seg *i_d_segs[32];
	struct rte_ml_buff_seg *o_q_segs[32];
	struct rte_ml_buff_seg *o_d_segs[32];
	struct rte_ml_buff_seg d_buf[4];
	struct rte_ml_buff_seg q_buf[4];
} ml_common_t;
typedef struct {
	uint64_t isize_q;
	uint64_t osize_q;
	uint64_t isize;
	uint64_t osize;
} ml_model_t;

/* ML global variables */
static ml_common_t ml_common;
static ml_model_t ml_model[ML_MAX_LAYERS];

void *
mrvl_ml_io_alloc(int model_id, enum buffer_type buff_type, uint64_t *size)
{
	struct rte_ml_model_info info;
	const struct rte_memzone *mz;
	char str[PATH_MAX];
	int ret = 0;

	/* get model info */
	ret = rte_ml_model_info_get(dev_ctx.dev_id, model_id, &info);
	if (ret != 0) {
		printf("Failed to get model info :\n");
		return NULL;
	}

	ml_model[model_id].isize = info.input_info->nb_elements * sizeof(float);
	ml_model[model_id].isize_q = info.input_info->size;
	ml_model[model_id].osize = info.output_info->nb_elements * sizeof(float);
	ml_model[model_id].osize_q = info.output_info->size;

	switch (buff_type) {
	case 0:
		snprintf(str, PATH_MAX, "model_q_input_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_model[model_id].isize_q, rte_socket_id(),
						 0, dev_ctx.dev_info.align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV, "Failed to create memzone for model quantize input:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_model[model_id].isize_q;
		return mz->addr;

	case 1:
		snprintf(str, PATH_MAX, "model_d_input_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_model[model_id].isize, rte_socket_id(), 0,
						 dev_ctx.dev_info.align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV,
				"Failed to create memzone for model dequantize input:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_model[model_id].isize;
		return mz->addr;

	case 2:
		snprintf(str, PATH_MAX, "model_q_output_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_model[model_id].osize_q, rte_socket_id(),
						 0, dev_ctx.dev_info.align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV,
				"Failed to create memzone for model quantize output:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_model[model_id].osize_q;
		return mz->addr;

	case 3:
		snprintf(str, PATH_MAX, "model_d_output_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_model[model_id].osize, rte_socket_id(), 0,
						 dev_ctx.dev_info.align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV,
				"Failed to create memzone for model dequantize output:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_model[model_id].osize;
		return mz->addr;
	}

	return NULL;
}

void
mrvl_ml_io_free(int model_id, enum buffer_type buff_type, void *addr)
{
	const struct rte_memzone *mz = NULL;
	char str[PATH_MAX];

	if (buff_type == input_quantize) {
		snprintf(str, PATH_MAX, "model_q_input_%d", model_id);
		mz = rte_memzone_lookup(str);
	} else if (buff_type == input_dequantize) {
		snprintf(str, PATH_MAX, "model_d_input_%d", model_id);
		mz = rte_memzone_lookup(str);
	} else if (buff_type == output_quantize) {
		snprintf(str, PATH_MAX, "model_q_output_%d", model_id);
		mz = rte_memzone_lookup(str);
	} else if (buff_type == output_dequantize) {
		snprintf(str, PATH_MAX, "model_d_output_%d", model_id);
		mz = rte_memzone_lookup(str);
	}
	if (mz != NULL) {
		if ((uint64_t)mz->addr == (uint64_t)addr)
			rte_memzone_free(mz);
	}
}

int
mrvl_ml_model_quantize(int model_id, void *dbuffer, void *qbuffer)
{
	int ret;

	ml_common.d_buf[0].addr = dbuffer;
	ml_common.d_buf[0].iova_addr = rte_mem_virt2iova(dbuffer);
	ml_common.d_buf[0].length = ml_model[model_id].isize;
	ml_common.d_buf[0].next = NULL;
	ml_common.i_d_segs[0] = &ml_common.d_buf[0];

	ml_common.q_buf[0].addr = qbuffer;
	ml_common.q_buf[0].iova_addr = rte_mem_virt2iova(qbuffer);
	ml_common.q_buf[0].length = ml_model[model_id].isize_q;
	ml_common.q_buf[0].next = NULL;
	ml_common.i_q_segs[0] = &ml_common.q_buf[0];

	/* Quantize input */
	ret = rte_ml_io_quantize(dev_ctx.dev_id, model_id, ml_common.i_d_segs, ml_common.i_q_segs);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Input Quantization failed,\n");
		return ret;
	}
	return ret;
}

int
mrvl_ml_model_dequantize(int model_id, void *qbuffer, void *dbuffer)
{
	int ret;

	ml_common.d_buf[1].addr = dbuffer;
	ml_common.d_buf[1].iova_addr = rte_mem_virt2iova(dbuffer);
	ml_common.d_buf[1].length = ml_model[model_id].osize;
	ml_common.d_buf[1].next = NULL;
	ml_common.o_d_segs[0] = &ml_common.d_buf[1];

	ret = rte_ml_io_dequantize(dev_ctx.dev_id, model_id, ml_common.o_q_segs,
				   ml_common.o_d_segs);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Dequantization failed for output\n");
		return ret;
	}
	return ret;
}

static int
parse_json(int argc, char *argv[], char *config_file)
{
	json_error_t json_error;
	json_t *json_object;
	json_t *json_array;
	json_t *json;

	int nb_args = 0;

	memset(eal_args, '\0', sizeof(eal_args));

	json = json_load_file(config_file, 0, &json_error);
	if (!json) {
		fprintf(stderr, "error: on line %d: %s\n", json_error.line, json_error.text);
		return -1;
	}

	strcpy(eal_args[nb_args], argv[nb_args]);
	nb_args++;

	json_object = json_object_get(json, "dev_type");
	if (strcmp(json_string_value(json_object), "pci") == 0) {
		strcpy(eal_args[nb_args], "-a");
		nb_args++;
	} else if (strcmp(json_string_value(json_object), "vdev") == 0) {
		strcpy(eal_args[nb_args], "--vdev=");
	} else {
		printf("Device recognition failed. Only PCI and vdev devices are supported\n");
		return -1;
	}

	json_object = json_object_get(json, "device_id");
	strcat(eal_args[nb_args], json_string_value(json_object));

	json_object = json_object_get(json, "attributes");
	for (int iter = 0; iter < json_array_size(json_object); iter++) {
		strcat(eal_args[nb_args], ",");
		json_array = json_array_get(json_object, iter);
		strcat(eal_args[nb_args], json_string_value(json_array));
	}
	nb_args++;

	json_object = json_object_get(json, "log_level");
	for (int iter = 0; iter < json_array_size(json_object); iter++) {
		json_array = json_array_get(json_object, iter);
		strcpy(eal_args[nb_args], json_string_value(json_array));
		nb_args++;
	}

	for (int i = 2; i < argc; i++) {
		strcpy(eal_args[nb_args], argv[i]);
		nb_args++;
	}

	eal_argv = malloc(nb_args * sizeof(char *));
	for (uint16_t k = 0; k <= nb_args; k++)
		eal_argv[k] = eal_args[k];

	return nb_args;
}

int
mrvl_ml_init(int argc, char *argv[])
{
	int ret;

	ret = mrvl_ml_init_mt(argc, argv, ML_DEFAULT_NUM_THREADS);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "mrvl_ml_init ... failed\n");
		return ret;
	}

	return ret;
}

int
mrvl_ml_init_mt(int argc, char *argv[], int num_threads)
{
	struct rte_ml_dev_qp_conf qp_conf;
	char *config_file;
	uint8_t dev_count;
	uint16_t qp_id;
	int ret;
	int i;

	config_file = getenv("LIBMLDPC_CONFIG_PATH");
	if (!config_file)
		config_file = LIBMLDPC_CONFIG_DEFAULT_PATH;
	RTE_LOG(INFO, MLDEV, "LIBMLDPC_CONFIG_PATH = %s\n", config_file);

	/* parse config file */
	eal_argc = parse_json(argc, argv, config_file);
	if (eal_argc < 0) {
		RTE_LOG(ERR, MLDEV, "Failed pasing config file: %s\n", config_file);
		return eal_argc;
	}

	for (i = 0; i < eal_argc; i++)
		RTE_LOG(ERR, MLDEV, "eal_args[%d] = %s\n", i, eal_args[i]);

	/* Init EAL */
	ret = rte_eal_init(eal_argc, eal_argv);
	if (ret < 0) {
		RTE_LOG(ERR, MLDEV, "rte_eal_init .. failed\n");
		return ret;
	}

	dev_count = rte_ml_dev_count();
	if (dev_count <= 0) {
		RTE_LOG(ERR, MLDEV, "No ML devices found. exit.\n");
		return dev_count;
	}

	/* Get socket and device info */
	dev_ctx.dev_id = RTE_MAX(0, dev_count - 1);
	ret = rte_ml_dev_info_get(dev_ctx.dev_id, &dev_ctx.dev_info);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Failed to get device info, dev_id = %d\n", dev_ctx.dev_id);
		return ret;
	}

	/* Configure ML devices, use only dev_ctx.dev_id = 0 */
	dev_ctx.dev_config.socket_id = rte_ml_dev_socket_id(dev_ctx.dev_id);
	dev_ctx.dev_config.nb_models = dev_ctx.dev_info.max_models;
	dev_ctx.dev_config.nb_queue_pairs = RTE_MIN(dev_ctx.dev_info.max_queue_pairs, num_threads);
	ret = rte_ml_dev_configure(dev_ctx.dev_id, &dev_ctx.dev_config);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Device configuration failed, dev_id = %d\n", dev_ctx.dev_id);
		return ret;
	}

	/* Create OP pool */
	dev_ctx.op_pool = rte_ml_op_pool_create("ml_op_pool", ML_OP_POOL_SIZE, 0, 0,
						dev_ctx.dev_config.socket_id);
	if (dev_ctx.op_pool == NULL) {
		RTE_LOG(ERR, MLDEV, "Failed to create op pool : %s\n", "ml_op_pool");
		return -rte_errno;
	}

	/* Setup queue pairs */
	qp_conf.nb_desc = ML_MAX_DESC_PER_QP;
	qp_conf.cb = NULL;
	for (qp_id = 0; qp_id < dev_ctx.dev_config.nb_queue_pairs; qp_id++) {
		ret = rte_ml_dev_queue_pair_setup(dev_ctx.dev_id, qp_id, &qp_conf,
						  dev_ctx.dev_config.socket_id);
		if (ret != 0) {
			RTE_LOG(ERR, MLDEV,
				"Device queue-pair setup failed, dev_id = %d, qp_id = %u\n",
				dev_ctx.dev_id, qp_id);
			return ret;
		}
	}

	/* Start device */
	ret = rte_ml_dev_start(dev_ctx.dev_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Device start failed, dev_id = %d\n", dev_ctx.dev_id);
		return ret;
	};

	return 0;
}

int
mrvl_ml_finish(void)
{
	int ret;

	/* Stop device */
	ret = rte_ml_dev_stop(dev_ctx.dev_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Device stop failed, mldev_id = %d\n", dev_ctx.dev_id);
		return ret;
	}

	/* Destroy op pool */
	rte_ml_op_pool_free(dev_ctx.op_pool);

	/* Close ML device */
	ret = rte_ml_dev_close(dev_ctx.dev_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Device close failed, mldev_id = %d\n", dev_ctx.dev_id);
		return ret;
	}

	/* Release memory */
	free(eal_argv);

	/* Clean up the EAL */
	return rte_eal_cleanup();
}

int
mrvl_ml_model_load(char *model_buffer, int model_size)
{

	const struct rte_memzone *mz;
	struct rte_ml_model_params params;
	uint16_t model_id;
	int ret = 0;

	mz = rte_memzone_reserve_aligned("model", model_size, rte_socket_id(), 0,
					 dev_ctx.dev_info.align_size);
	if (mz == NULL) {
		fprintf(stderr, "Failed to create model memzone: \n");
		return -1;
	}

	params.addr = mz->addr;
	params.size = model_size;

	memcpy(params.addr, model_buffer, model_size);

	/*load the model */
	ret = rte_ml_model_load(dev_ctx.dev_id, &params, &model_id);
	if (ret != 0) {
		fprintf(stderr, "Error loading model\n");
		return ret;
	}

	/* Start model */
	ret = rte_ml_model_start(dev_ctx.dev_id, model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model start failed, model_id = %d\n", model_id);
		printf("return val is %d\n", ret);
		return ret;
	}

	rte_memzone_free(mz);

	return model_id;
}

static void
print_line(uint16_t len)
{
	uint16_t i;

	for (i = 0; i < len; i++)
		printf("-");

	printf("\n");
}

int j = 0;
int
ml_inference_get_stats(int model_id)
{
	struct test_inference ti;
	struct test_inference *t = &ti;
	int ret = 0;
	int i;
	const struct rte_memzone *mz;
	uint64_t total_avg_time = 0;

	t->xstats_size = rte_ml_dev_xstats_names_get(dev_ctx.dev_id, RTE_ML_DEV_XSTATS_MODEL,
						     model_id, NULL, 0);
	if (t->xstats_size >= 0) {
		/* allocate for xstats_map and values */
		mz = rte_memzone_reserve_aligned(
			"ml_xstats_map", t->xstats_size * sizeof(struct rte_ml_dev_xstats_map), -1,
			0, 0);
		t->xstats_map = mz->addr;
		if (t->xstats_map == NULL) {
			ret = -ENOMEM;
			goto error;
		}
		mz = rte_memzone_reserve_aligned("ml_xstats_values",
						 t->xstats_size * sizeof(uint64_t), -1, 0, 0);
		t->xstats_values = mz->addr;
		if (t->xstats_values == NULL) {
			ret = -ENOMEM;
			goto error;
		}
		ret = rte_ml_dev_xstats_names_get(dev_ctx.dev_id, RTE_ML_DEV_XSTATS_MODEL, model_id,
						  t->xstats_map, t->xstats_size);
		if (ret != t->xstats_size) {
			printf("Unable to get xstats names, ret = %d\n", ret);
			ret = -1;
			goto error;
		}
		for (i = 0; i < t->xstats_size; i++)
			rte_ml_dev_xstats_get(dev_ctx.dev_id, RTE_ML_DEV_XSTATS_MODEL, model_id,
					      &t->xstats_map[i].id, &t->xstats_values[i], 1);
	}
	if (j == 0) {
		printf("\n");
		print_line(80);
		printf(" ML Device Extended Statistics\n");
		print_line(80);
		for (i = 0; i < t->xstats_size; i = i + 6) {
			printf(" %-64s = %" PRIu64 "\n", t->xstats_map[i].name,
			       t->xstats_values[i]);
			total_avg_time += t->xstats_values[i];
		}
		printf("Total average time is %ld\n", total_avg_time);

		print_line(80);
		ret = 0;
		j++;
	}

	ret = 0;
error:
	/* release buffers */
	mz = rte_memzone_lookup("ml_xstats_map");
	if (mz != NULL)
		rte_memzone_free(mz);
	mz = rte_memzone_lookup("ml_xstats_values");
	if (mz != NULL)
		rte_memzone_free(mz);

	return ret;
}

int
mrvl_ml_model_unload(int model_id)
{
	int ret = 0;

	ret = ml_inference_get_stats(model_id);
	if (ret != 0) {
		printf("Error in getting stat\n");
		return ret;
	}

	/* Stop model */
	ret = rte_ml_model_stop(dev_ctx.dev_id, model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model stop failed, mldev_id = %d\n", model_id);
		return ret;
	}

	ret = rte_ml_model_unload(dev_ctx.dev_id, model_id);
	if (ret != 0) {
		fprintf(stderr, "Error loading model : \n");
		return ret;
	}
	return ret;
}

int
mrvl_ml_model_run(struct run_args *run_arg)
{
	int ret = 0;

	ret = mrvl_ml_model_run_mt(run_arg, 0);
	if (ret != 0) {
		printf("mrvl_ml_model_run() Failed\n");
		return -1;
	}
	return ret;
}

int
mrvl_ml_model_run_mt(struct run_args *run_arg, int thread_id)
{

	struct rte_ml_op *op;
	uint32_t burst_enq = 1;
	uint32_t total_enq = 0;
	uint32_t burst_deq;
	uint32_t total_deq = 0;
	struct rte_ml_op_error error;

	ml_common.q_buf[1].addr = run_arg->out_buf;
	ml_common.q_buf[1].iova_addr = rte_mem_virt2iova(run_arg->out_buf);
	ml_common.q_buf[1].length = ml_model[run_arg->model_id].osize_q;
	ml_common.q_buf[1].next = NULL;
	ml_common.o_q_segs[0] = &ml_common.q_buf[1];

	if (rte_mempool_get(dev_ctx.op_pool, (void **)&op) != 0)
		return -1;

enqueue_req:
	if (burst_enq == 1) {
		/* Update ML Op */
		op->model_id = run_arg->model_id;
		op->nb_batches = run_arg->num_batches;
		op->mempool = dev_ctx.op_pool;

		op->input = ml_common.i_q_segs;
		op->output = ml_common.o_q_segs;
	}
	burst_enq = rte_ml_enqueue_burst(dev_ctx.dev_id, thread_id, &op, 1);
	if (burst_enq == 0)
		goto enqueue_req;

	if (likely(burst_enq == 1)) {
		total_enq += burst_enq;

		if (unlikely(op->status == RTE_ML_OP_STATUS_ERROR)) {
			rte_ml_op_error_get(dev_ctx.dev_id, op, &error);
			RTE_LOG(ERR, MLDEV, "error_code = 0x%016lx, error_message = %s\n",
				error.errcode, error.message);
			rte_mempool_put(dev_ctx.op_pool, op);
			return error.errcode;
		} else {
			rte_mempool_put(dev_ctx.op_pool, op);
		}
	}

dequeue_req:
	/* dequeue burst */
	burst_deq = rte_ml_dequeue_burst(dev_ctx.dev_id, thread_id, &op, 1);
	if (likely(burst_deq == 1)) {
		total_deq += burst_deq;

		if (unlikely(op->status == RTE_ML_OP_STATUS_ERROR)) {
			rte_ml_op_error_get(dev_ctx.dev_id, op, &error);
			RTE_LOG(ERR, MLDEV, "error_code = 0x%016lx, error_message = %s\n",
				error.errcode, error.message);
			rte_mempool_put(dev_ctx.op_pool, op);
			return error.errcode;
		}
	} else if (burst_deq == 0)
		goto dequeue_req;

	return 0;
}
