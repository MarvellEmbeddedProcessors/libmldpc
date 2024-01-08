/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2023 Marvell.
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

#include <mldpc.h>

/* constants */
#define ML_DEFAULT_NUM_THREADS 1
#define ML_MAX_EAL_ARGS	       64
#define ML_OP_POOL_SIZE	       1024
#define ML_MAX_DESC_PER_QP     1
#define ML_MAX_MODELS	       64

/* log stream */
#define RTE_LOGTYPE_MLDEV RTE_LOGTYPE_USER1

/* default EAL config */
#define LIBMLDPC_CONFIG_DEFAULT_PATH "/usr/share/mldpc/config.json"

/* device context */
typedef struct ml_dev_ctx {
	int16_t dev_id;
	struct rte_ml_dev_info dev_info;
	struct rte_ml_dev_config dev_config;
	struct rte_mempool *op_pool;
} ml_dev_ctx_t;

/* model context */
typedef struct ml_model_ctx {
	struct rte_ml_model_info model_info;

	uint64_t input_size_q;
	uint64_t output_size_q;
	uint64_t input_size_d;
	uint64_t output_size_d;

	struct rte_ml_buff_seg input_seg_q;
	struct rte_ml_buff_seg input_seg_d;
	struct rte_ml_buff_seg output_seg_q;
	struct rte_ml_buff_seg output_seg_d;

	struct rte_ml_buff_seg *input_seg_array_d;
	struct rte_ml_buff_seg *input_seg_array_q;
	struct rte_ml_buff_seg *output_seg_array_q;
	struct rte_ml_buff_seg *output_seg_array_d;
} ml_model_ctx_t;

ml_dev_ctx_t dev_ctx;
ml_model_ctx_t model_ctx[ML_MAX_MODELS];

/* EAL variables */
int eal_argc;
char eal_args[ML_MAX_EAL_ARGS][PATH_MAX];
char **eal_argv;

static int
parse_json(int argc, char *argv[], char *config_file)
{
	json_error_t json_error;
	json_t *json_object;
	json_t *json_array;
	json_t *json;

	int nb_args = 0;

	(void)argc;

	memset(eal_args, '\0', sizeof(eal_args));

	json = json_load_file(config_file, 0, &json_error);
	if (!json) {
		fprintf(stderr, "error: on line %d: %s\n", json_error.line, json_error.text);
		return -1;
	}

	strcpy(eal_args[nb_args], argv[nb_args]);
	nb_args++;

	json_object = json_object_get(json, "lcores");
	strcpy(eal_args[nb_args], "--lcores=");
	strcat(eal_args[nb_args], json_string_value(json_object));
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
	for (uint64_t iter = 0; iter < json_array_size(json_object); iter++) {
		strcat(eal_args[nb_args], ",");
		json_array = json_array_get(json_object, iter);
		strcat(eal_args[nb_args], json_string_value(json_array));
	}
	nb_args++;

	json_object = json_object_get(json, "log_level");
	for (uint64_t iter = 0; iter < json_array_size(json_object); iter++) {
		json_array = json_array_get(json_object, iter);
		strcpy(eal_args[nb_args], json_string_value(json_array));
		nb_args++;
	}

	eal_argv = malloc(nb_args * sizeof(char *));
	for (uint16_t k = 0; k <= nb_args; k++)
		eal_argv[k] = eal_args[k];

	return nb_args;
}

static void
print_line(uint16_t len)
{
	uint16_t i;

	for (i = 0; i < len; i++)
		printf("-");

	printf("\n");
}

static int
ml_inference_get_stats(int model_id)
{
	struct rte_ml_dev_xstats_map *xstats_map;
	const struct rte_memzone *mz;
	uint64_t *xstats_values;
	int xstats_size;
	int ret = 0;
	int i;

	xstats_size = rte_ml_dev_xstats_names_get(dev_ctx.dev_id, RTE_ML_DEV_XSTATS_MODEL, model_id,
						  NULL, 0);
	if (xstats_size >= 0) {
		/* allocate for xstats_map and values */
		mz = rte_memzone_reserve_aligned("ml_xstats_map",
						 xstats_size * sizeof(struct rte_ml_dev_xstats_map),
						 -1, 0, 0);
		xstats_map = mz->addr;
		if (xstats_map == NULL) {
			ret = -ENOMEM;
			goto error;
		}
		mz = rte_memzone_reserve_aligned("ml_xstats_values", xstats_size * sizeof(uint64_t),
						 -1, 0, 0);
		xstats_values = mz->addr;
		if (xstats_values == NULL) {
			ret = -ENOMEM;
			goto error;
		}
		ret = rte_ml_dev_xstats_names_get(dev_ctx.dev_id, RTE_ML_DEV_XSTATS_MODEL, model_id,
						  xstats_map, xstats_size);
		if (ret != xstats_size) {
			printf("Unable to get xstats names, ret = %d\n", ret);
			ret = -1;
			goto error;
		}

		for (i = 0; i < xstats_size; i++)
			rte_ml_dev_xstats_get(dev_ctx.dev_id, RTE_ML_DEV_XSTATS_MODEL, model_id,
					      &xstats_map[i].id, &xstats_values[i], 1);

		printf("\n");
		print_line(80);
		printf(" Inference Statistics, model_id = %u\n", model_id);
		print_line(80);
		for (i = 0; i < xstats_size; i++)
			printf(" %-64s = %" PRIu64 "\n", xstats_map[i].name, xstats_values[i]);
		print_line(80);
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
mrvl_ml_model_load(char *buffer, int size)
{
	struct rte_ml_model_params params;
	uint16_t model_id;
	int ret;

	/* Load model */
	params.addr = buffer;
	params.size = size;
	ret = rte_ml_model_load(dev_ctx.dev_id, &params, &model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model load failed, dev_id = %d, buffer = %p\n", dev_ctx.dev_id,
			buffer);
		return ret;
	}

	/* Start model */
	ret = rte_ml_model_start(dev_ctx.dev_id, model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model start failed, dev_id = %d, model_id = %d\n",
			dev_ctx.dev_id, model_id);
		return ret;
	}

	/* Get model info */
	ret = rte_ml_model_info_get(dev_ctx.dev_id, model_id, &model_ctx[model_id].model_info);
	if (ret != 0) {
		printf("Failed to get model info, dev_id = %d, model_id = %u\n", dev_ctx.dev_id,
		       model_id);
		return ret;
	}

	/* initialize model context */
	model_ctx[model_id].input_size_d =
		model_ctx[model_id].model_info.input_info->nb_elements * sizeof(float);
	model_ctx[model_id].input_seg_d.addr = NULL;
	model_ctx[model_id].input_seg_d.iova_addr = RTE_BAD_IOVA;
	model_ctx[model_id].input_seg_d.length = model_ctx[model_id].input_size_d;
	model_ctx[model_id].input_seg_d.next = NULL;
	model_ctx[model_id].input_seg_array_d = &model_ctx[model_id].input_seg_d;

	model_ctx[model_id].input_size_q = model_ctx[model_id].model_info.input_info->size;
	model_ctx[model_id].input_seg_q.addr = NULL;
	model_ctx[model_id].input_seg_q.iova_addr = RTE_BAD_IOVA;
	model_ctx[model_id].input_seg_q.length = model_ctx[model_id].input_size_d;
	model_ctx[model_id].input_seg_q.next = NULL;
	model_ctx[model_id].input_seg_array_q = &model_ctx[model_id].input_seg_q;

	model_ctx[model_id].output_size_q = model_ctx[model_id].model_info.output_info->size;
	model_ctx[model_id].output_seg_q.addr = NULL;
	model_ctx[model_id].output_seg_q.iova_addr = RTE_BAD_IOVA;
	model_ctx[model_id].output_seg_q.length = model_ctx[model_id].output_size_q;
	model_ctx[model_id].output_seg_q.next = NULL;
	model_ctx[model_id].output_seg_array_q = &model_ctx[model_id].output_seg_q;

	model_ctx[model_id].output_size_d =
		model_ctx[model_id].model_info.output_info->nb_elements * sizeof(float);
	model_ctx[model_id].output_seg_d.addr = NULL;
	model_ctx[model_id].output_seg_d.iova_addr = RTE_BAD_IOVA;
	model_ctx[model_id].output_seg_d.length = model_ctx[model_id].output_size_d;
	model_ctx[model_id].output_seg_d.next = NULL;
	model_ctx[model_id].output_seg_array_d = &model_ctx[model_id].output_seg_d;

	return model_id;
}

int
mrvl_ml_model_unload(int model_id)
{
	int ret;

	ret = ml_inference_get_stats(model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Error getting inference stats, dev_id = %d, model_id = %u\n",
			dev_ctx.dev_id, model_id);
		return ret;
	}

	/* Stop model */
	ret = rte_ml_model_stop(dev_ctx.dev_id, model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model stop failed, dev_id = %d, model_id = %u\n",
			dev_ctx.dev_id, model_id);
		return ret;
	}

	/* Unload model */
	ret = rte_ml_model_unload(dev_ctx.dev_id, model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model unload failed, dev_id = %d, model_id = %u\n",
			dev_ctx.dev_id, model_id);
		return ret;
	}

	return 0;
}

void *
mrvl_ml_io_alloc(int model_id, enum buffer_type buff_type, uint64_t *size)
{
	const struct rte_memzone *mz;
	char str[PATH_MAX];
	uint64_t lcl_size;

	switch (buff_type) {
	case input_quantize:
		snprintf(str, PATH_MAX, "model_q_input_%d", model_id);
		lcl_size = model_ctx[model_id].input_size_q;
		break;
	case input_dequantize:
		snprintf(str, PATH_MAX, "model_d_input_%d", model_id);
		lcl_size = model_ctx[model_id].input_size_d;
		break;
	case output_quantize:
		snprintf(str, PATH_MAX, "model_q_output_%d", model_id);
		lcl_size = model_ctx[model_id].output_size_q;
		break;
	case output_dequantize:
		snprintf(str, PATH_MAX, "model_d_output_%d", model_id);
		lcl_size = model_ctx[model_id].output_size_d;
		break;
	default:
		RTE_LOG(ERR, MLDEV, "Invalid buffer_type = %d\n", buff_type);
		return NULL;
	}

	mz = rte_memzone_reserve_aligned(str, lcl_size, dev_ctx.dev_config.socket_id, 0,
					 dev_ctx.dev_info.align_size);
	if (mz == NULL) {
		RTE_LOG(ERR, MLDEV,
			"Failed to create memzone for I/O data, dev_id = %d, model_id = %d, buffer_type = %d\n",
			dev_ctx.dev_id, model_id, buff_type);
		return NULL;
	}

	if (size != NULL)
		*size = lcl_size;

	return mz->addr;
}

void
mrvl_ml_io_free(int model_id, enum buffer_type buff_type, void *addr)
{
	const struct rte_memzone *mz = NULL;
	char str[PATH_MAX];

	switch (buff_type) {
	case input_quantize:
		snprintf(str, PATH_MAX, "model_q_input_%d", model_id);
		break;
	case input_dequantize:
		snprintf(str, PATH_MAX, "model_d_input_%d", model_id);
		break;
	case output_quantize:
		snprintf(str, PATH_MAX, "model_q_output_%d", model_id);
		break;
	case output_dequantize:
		snprintf(str, PATH_MAX, "model_d_output_%d", model_id);
		break;
	default:
		RTE_LOG(ERR, MLDEV, "Invalid buffer_type = %d\n", buff_type);
		return;
	}

	mz = rte_memzone_lookup(str);
	if (mz != NULL) {
		if ((uint64_t)mz->addr == (uint64_t)addr)
			rte_memzone_free(mz);
	}
}

int
mrvl_ml_model_quantize(int model_id, void *dbuffer, void *qbuffer)
{
	int ret;

	model_ctx[model_id].input_seg_d.addr = dbuffer;
	model_ctx[model_id].input_seg_d.iova_addr = rte_mem_virt2iova(dbuffer);

	model_ctx[model_id].input_seg_q.addr = qbuffer;
	model_ctx[model_id].input_seg_q.iova_addr = rte_mem_virt2iova(qbuffer);

	/* Quantize input */
	ret = rte_ml_io_quantize(dev_ctx.dev_id, model_id, &model_ctx[model_id].input_seg_array_d,
				 &model_ctx[model_id].input_seg_array_q);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Input Quantization failed, model_id = %u\n", model_id);
		return ret;
	}

	return 0;
}

int
mrvl_ml_model_dequantize(int model_id, void *qbuffer, void *dbuffer)
{
	int ret;

	model_ctx[model_id].output_seg_q.addr = qbuffer;
	model_ctx[model_id].output_seg_q.iova_addr = rte_mem_virt2iova(qbuffer);

	model_ctx[model_id].output_seg_d.addr = dbuffer;
	model_ctx[model_id].output_seg_d.iova_addr = rte_mem_virt2iova(dbuffer);

	ret = rte_ml_io_dequantize(dev_ctx.dev_id, model_id,
				   &model_ctx[model_id].output_seg_array_q,
				   &model_ctx[model_id].output_seg_array_d);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Output Dequantization failed, model_id = %u\n", model_id);
		return ret;
	}

	return 0;
}

int
mrvl_ml_model_run(int model_id, void *input_buffer, void *output_buffer, int num_batches)
{
	int ret = 0;

	ret = mrvl_ml_model_run_mt(model_id, input_buffer, output_buffer, num_batches, 0);
	if (ret != 0) {
		printf("mrvl_ml_model_run() Failed\n");
		return -1;
	}
	return ret;
}

int
mrvl_ml_model_run_mt(int model_id, void *input_buffer, void *output_buffer, int num_batches,
		     int thread_id)
{
	struct rte_ml_op_error error;
	uint16_t enqueued = 0;
	uint16_t dequeued = 0;
	struct rte_ml_op *op;
	int ret = 0;

	model_ctx[model_id].input_seg_q.addr = input_buffer;
	model_ctx[model_id].input_seg_q.iova_addr = rte_mem_virt2iova(input_buffer);

	model_ctx[model_id].output_seg_q.addr = output_buffer;
	model_ctx[model_id].output_seg_q.iova_addr = rte_mem_virt2iova(output_buffer);

	if (rte_mempool_get(dev_ctx.op_pool, (void **)&op) != 0)
		return -1;

	op->model_id = model_id;
	op->nb_batches = num_batches;
	op->mempool = dev_ctx.op_pool;
	op->input = &model_ctx[model_id].input_seg_array_q;
	op->output = &model_ctx[model_id].output_seg_array_q;

enqueue_req:
	enqueued = rte_ml_enqueue_burst(dev_ctx.dev_id, thread_id, &op, 1);
	if (unlikely(enqueued == 0))
		goto enqueue_req;

dequeue_req:
	dequeued = rte_ml_dequeue_burst(dev_ctx.dev_id, thread_id, &op, 1);
	if (likely(dequeued == 1)) {
		if (unlikely(op->status == RTE_ML_OP_STATUS_ERROR)) {
			rte_ml_op_error_get(dev_ctx.dev_id, op, &error);
			RTE_LOG(ERR, MLDEV, "error_code = 0x%016lx, error_message = %s\n",
				error.errcode, error.message);
			ret = -1;
		}

		rte_mempool_put(dev_ctx.op_pool, op);
	} else {
		goto dequeue_req;
	}

	return ret;
}
