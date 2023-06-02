/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2022 Marvell.
 */

#include <getopt.h>
#include <stdio.h>

#include <jansson.h>
#include <rte_eal.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <rte_mldev.h>

#include "tvm_dpdk_mldev.h"

/* ML constants */
#define ML_OP_POOL_SIZE 1024
#define MAX_ARGS	64
#define MAX_ARGS_LEN	128

#define RTE_LOGTYPE_MLDEV RTE_LOGTYPE_USER1

#define MIN(x, y) ((x < y) ? x : y)

/* ML model variables structure */
typedef struct {
	uint8_t dev_id;
	uint8_t model_id;

	uint16_t min_align_size;

	struct rte_mempool *mp;

	uint8_t qp_id;
	struct rte_ml_dev_qp_conf qp_conf;

	uint64_t isize;

	uint64_t isize_q;

	uint64_t osize_q;

	uint64_t osize;
} ml_model_vars_t;

/* ML global variables */
static ml_model_vars_t ml_models;

void *
mrvl_ml_io_alloc(int model_id, enum buffer_type buff_type, int num_batches, uint64_t *size)
{
	const struct rte_memzone *mz;
	char str[PATH_MAX];
	int ret = 0;

	/* get input buffer size */
	ret = rte_ml_io_input_size_get(ml_models.dev_id, model_id, num_batches, &ml_models.isize_q,
				       &ml_models.isize);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Error getting input size\n");
		return NULL;
	}

	/* get output buffer size */
	ret = rte_ml_io_output_size_get(ml_models.dev_id, model_id, num_batches, &ml_models.osize_q,
					&ml_models.osize);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Error getting output size\n");
		return NULL;
	}

	switch (buff_type) {
	case 0:
		snprintf(str, PATH_MAX, "model_q_input_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_models.isize_q, rte_socket_id(), 0,
						 ml_models.min_align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV, "Failed to create memzone for model quantize input:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_models.isize_q;
		return mz->addr;

	case 1:
		snprintf(str, PATH_MAX, "model_d_input_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_models.isize, rte_socket_id(), 0,
						 ml_models.min_align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV,
				"Failed to create memzone for model dequantize input:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_models.isize;
		return mz->addr;

	case 2:
		snprintf(str, PATH_MAX, "model_q_output_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_models.osize_q, rte_socket_id(), 0,
						 ml_models.min_align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV,
				"Failed to create memzone for model quantize output:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_models.osize_q;
		return mz->addr;

	case 3:
		snprintf(str, PATH_MAX, "model_d_output_%d", model_id);
		mz = rte_memzone_reserve_aligned(str, ml_models.osize, rte_socket_id(), 0,
						 ml_models.min_align_size);
		if (mz == NULL) {
			RTE_LOG(ERR, MLDEV,
				"Failed to create memzone for model dequantize output:\n");
			return NULL;
		}
		if (size != NULL)
			*size = ml_models.osize;
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
mrvl_ml_model_quantize(int model_id, int num_batches, void *dbuffer, void *qbuffer)
{

	int ret;

	/* Quantize input */
	ret = rte_ml_io_quantize(ml_models.dev_id, model_id, num_batches, dbuffer, qbuffer);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Input Quantization failed,\n");
		return ret;
	}
	return ret;
}

int
mrvl_ml_model_dequantize(int model_id, int num_batches, void *qbuffer, void *dbuffer)
{

	int ret;

	ret = rte_ml_io_dequantize(ml_models.dev_id, model_id, num_batches, qbuffer, dbuffer);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Dequantization failed for output\n");
		return ret;
	}
	return ret;
}

char **v_args;
char c_args[MAX_ARGS][MAX_ARGS_LEN];

static int
parse_json(int argc, char *argv[])
{
	json_error_t error;
	json_t *parsed_json, *j_arr_data;
	json_t *j_obj;
	int arg_count = 0;

	memset(c_args, '\0', sizeof(c_args));

	if (argv[1] != NULL) {
		parsed_json = json_load_file(argv[1], 0, &error);
		if (!parsed_json) {
			fprintf(stderr, "error: on line %d: %s\n", error.line, error.text);
			return -1;
		}
	} else {
		printf("Error... Required config file to pass\n");
		return -1;
	}

	strcpy(c_args[arg_count], argv[arg_count]);
	arg_count++;

	j_obj = json_object_get(parsed_json, "dev_type");
	if (strcmp(json_string_value(j_obj), "pci") == 0) {
		strcpy(c_args[arg_count], "-a");
		arg_count++;
	} else if (strcmp(json_string_value(j_obj), "vdev") == 0) {
		strcpy(c_args[arg_count], "--vdev=");
	} else {
		printf("Device recognition failed. Only PCI and vdev devices are supported\n");
		return -1;
	}

	j_obj = json_object_get(parsed_json, "device_id");
	strcat(c_args[arg_count], json_string_value(j_obj));

	j_obj = json_object_get(parsed_json, "attributes");
	for (int iter = 0; iter < json_array_size(j_obj); iter++) {
		strcat(c_args[arg_count], ",");
		j_arr_data = json_array_get(j_obj, iter);
		strcat(c_args[arg_count], json_string_value(j_arr_data));
	}
	arg_count++;

	j_obj = json_object_get(parsed_json, "log_level");
	for (int iter = 0; iter < json_array_size(j_obj); iter++) {
		j_arr_data = json_array_get(j_obj, iter);
		strcpy(c_args[arg_count], json_string_value(j_arr_data));
		arg_count++;
	}

	for (int i = 2; i < argc; i++) {
		strcpy(c_args[arg_count], argv[i]);
		arg_count++;
	}
	v_args = malloc(arg_count * sizeof(char *));
	for (uint16_t k = 0; k <= arg_count; k++)
		v_args[k] = c_args[k];

	return arg_count;
}

int
mrvl_ml_init(int argc, char *argv[])
{
	int ret = 0;

	ret = mrvl_ml_init_mt(argc, argv, 1);
	if (ret != 0) {
		printf("mrvl_ml_init() Failed...");
		return ret;
	}

	return ret;
}

int
mrvl_ml_init_mt(int argc, char *argv[], int num_threads)
{
	int ret = 0;
	uint8_t dev_count;
	struct rte_ml_dev_info dev_info;
	struct rte_ml_dev_config ml_config;

	argc = parse_json(argc, argv);
	if (argc < 0) {
		printf("Json parse Failed...\n");
		return argc;
	}

	for (uint16_t i = 0; i < argc; i++)
		printf("v_args[%d] = %s\n", i, v_args[i]);

	/* Init EAL */
	printf("argc is %d\n", argc);
	ret = rte_eal_init(argc, v_args);
	if (ret < 0) {
		printf("rte_eal_init() api failed...\n");
		return ret;
	}

	dev_count = rte_ml_dev_count();
	if (dev_count <= 0) {
		fprintf(stderr, "No ML devices found. exit.\n");
		return dev_count;
	}

	/* Get socket and device info */
	ml_models.dev_id = dev_count - 1;
	ret = rte_ml_dev_info_get(ml_models.dev_id, &dev_info);
	if (ret != 0) {
		fprintf(stderr, "Failed to get device info, ml_models.dev_id = %d\n",
			ml_models.dev_id);
		return ret;
	}

	/* Configure ML devices, use only ml_models.dev_id = 0 */
	ml_models.min_align_size = dev_info.min_align_size;
	ml_config.socket_id = rte_ml_dev_socket_id(ml_models.dev_id);
	ml_config.nb_models = dev_info.max_models;
	ml_config.nb_queue_pairs = MIN(dev_info.max_queue_pairs, num_threads);
	ret = rte_ml_dev_configure(ml_models.dev_id, &ml_config);
	if (ret != 0) {
		fprintf(stderr, "Device configuration failed, ml_models.dev_id = %d\n",
			ml_models.dev_id);
		return ret;
	}

	/* Create OP mempool */
	ml_models.mp =
		rte_ml_op_pool_create("ml_op_pool", ML_OP_POOL_SIZE, 0, 0, ml_config.socket_id);
	if (ml_models.mp == NULL) {
		RTE_LOG(ERR, MLDEV, "Failed to create op pool : %s\n", "ml_op_pool");
		return -1;
	}

	/* setup queue pairs */
	ml_models.qp_conf.nb_desc = 1;
	ml_models.qp_conf.cb = NULL;
	for (ml_models.qp_id = 0; ml_models.qp_id < ml_config.nb_queue_pairs; ml_models.qp_id++) {
		ret = rte_ml_dev_queue_pair_setup(ml_models.dev_id, ml_models.qp_id,
						  &ml_models.qp_conf, ml_config.socket_id);
		if (ret != 0) {
			RTE_LOG(ERR, MLDEV, "Device queue-pair setup failed, mldev_id = %d\n",
				ml_models.dev_id);
			return ret;
		}
	}

	/* Start device */
	ret = rte_ml_dev_start(ml_models.dev_id);
	if (ret != 0) {
		fprintf(stderr, "Device start failed, ml_models.dev_id = %d\n", ml_models.dev_id);
		return ret;
	};

	return ret;
}

int
mrvl_ml_model_load(char *model_buffer, int model_size, int num_batches)
{

	const struct rte_memzone *mz;
	struct rte_ml_model_params params;
	int16_t model_id;
	int ret = 0;

	mz = rte_memzone_reserve_aligned("model", model_size, rte_socket_id(), 0,
					 ml_models.min_align_size);
	if (mz == NULL) {
		fprintf(stderr, "Failed to create model memzone: \n");
		return -1;
	}

	params.addr = mz->addr;
	params.size = model_size;

	memcpy(params.addr, model_buffer, model_size);

	/*load the model */
	ret = rte_ml_model_load(ml_models.dev_id, &params, &model_id);
	if (ret != 0) {
		fprintf(stderr, "Error loading model\n");
		return ret;
	}

	/* Start model */
	ret = rte_ml_model_start(ml_models.dev_id, model_id);
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
ml_inference_get_stats()
{
	struct test_inference ti;
	struct test_inference *t = &ti;
	int ret;
	int i;
	const struct rte_memzone *mz;
	uint64_t total_avg_time = 0;

	t->xstats_size = rte_ml_dev_xstats_names_get(ml_models.dev_id, NULL, 0);
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
		ret = rte_ml_dev_xstats_names_get(ml_models.dev_id, t->xstats_map, t->xstats_size);
		if (ret != t->xstats_size) {
			printf("Unable to get xstats names, ret = %d\n", ret);
			ret = -1;
			goto error;
		}
		for (i = 0; i < t->xstats_size; i++)
			rte_ml_dev_xstats_get(ml_models.dev_id, &t->xstats_map[i].id,
					      &t->xstats_values[i], 1);
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
	int err = 0;

	ret = ml_inference_get_stats();
	if (ret != 0) {
		printf("Error in getting stat\n");
		return ret;
	}

	/* Stop model */
	ret = rte_ml_model_stop(ml_models.dev_id, model_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Model stop failed, mldev_id = %d\n", model_id);
		return ret;
	}

	ret = rte_ml_model_unload(ml_models.dev_id, model_id);
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

	struct rte_ml_op *op = NULL;
	uint32_t burst_enq = 1;
	uint32_t total_enq = 0;
	uint32_t burst_deq;
	uint32_t total_deq = 0;
	struct rte_ml_op_error error;

	while (total_enq < run_arg->repetitions) {
	enqueue_req:
		if (burst_enq == 1) {
			rte_mempool_get(ml_models.mp, (void **)&op);
			if (unlikely(op == NULL))
				goto enqueue_req;

			/* Update ML Op */
			op->model_id = run_arg->model_id;
			op->nb_batches = run_arg->num_batches;
			op->mempool = ml_models.mp;

			op->input.addr = run_arg->input_buf;
			op->input.next = NULL;

			op->output.addr = run_arg->out_buf;
			op->output.next = NULL;
			if (run_arg->mdl_type == TVM) {
				op->input.length = ml_models.isize_q;
				op->output.length = ml_models.osize_q;
			} else if (run_arg->mdl_type == MLIP) {
				op->input.length = ml_models.isize;
				op->output.length = ml_models.osize;
			}
		}
		burst_enq = rte_ml_enqueue_burst(ml_models.dev_id, thread_id, &op, 1);

		if (likely(burst_enq == 1)) {
			total_enq += burst_enq;

			if (unlikely(op->status == RTE_ML_OP_STATUS_ERROR)) {
				rte_ml_op_error_get(ml_models.dev_id, op, &error);
				RTE_LOG(ERR, MLDEV, "error_code = 0x%016lx, error_message = %s\n",
					error.errcode, error.message);
				rte_mempool_put(ml_models.mp, op);
				return error.errcode;
			}
		}
	}

	while (total_deq < run_arg->repetitions) {
		/* dequeue burst */
		burst_deq = rte_ml_dequeue_burst(ml_models.dev_id, thread_id, &op, 1);
		if (likely(burst_deq == 1)) {
			total_deq += burst_deq;

			if (unlikely(op->status == RTE_ML_OP_STATUS_ERROR)) {
				rte_ml_op_error_get(ml_models.dev_id, op, &error);
				RTE_LOG(ERR, MLDEV, "error_code = 0x%016lx, error_message = %s\n",
					error.errcode, error.message);
				rte_mempool_put(ml_models.mp, op);
				return error.errcode;
			}
			rte_mempool_put(ml_models.mp, op);
		}
	}
	return 0;
}

int
mrvl_ml_model_finish()
{

	int ret = 0;

	/* Stop device */
	ret = rte_ml_dev_stop(ml_models.dev_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Device stop failed, mldev_id = %d\n", ml_models.dev_id);
		return ret;
	}

	/* Destroy op pool */
	rte_ml_op_pool_free(ml_models.mp);

	/* Close ML device */
	ret = rte_ml_dev_close(ml_models.dev_id);
	if (ret != 0) {
		RTE_LOG(ERR, MLDEV, "Device close failed, mldev_id = %d\n", ml_models.dev_id);
		return ret;
	}

	/* clean up the EAL */
	rte_eal_cleanup();
	return ret;
}
