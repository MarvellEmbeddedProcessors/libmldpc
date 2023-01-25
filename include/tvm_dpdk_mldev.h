/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2022 Marvell.
 */

#ifndef __VELOX_MLINTF_H__
#define __VELOX_MLINTF_H__

#include <stdint.h>
#include <stdio.h>
enum buffer_type { input_quantize = 0, input_dequantize, output_quantize, output_dequantize };

enum model_type { TVM = 0, MLIP };

struct test_inference {
	struct rte_ml_dev_xstats_map *xstats_map;
	uint64_t *xstats_values;
	int xstats_size;
} __rte_cache_aligned;

struct run_args {
	int model_id;
	void *input_buf;
	void *out_buf;
	int num_batches;
	enum model_type mdl_type;
	uint64_t repetitions;
};

void *mrvl_ml_io_alloc(int model_id, enum buffer_type buff_type, int num_batches, uint64_t *size);
void mrvl_ml_io_free(int model_id, enum buffer_type buff_type, void *addr);
int mrvl_ml_model_quantize(int model_id, int num_batches, void *dbuffer, void *qbuffer);
int mrvl_ml_model_dequantize(int model_id, int num_batches, void *qbuffer, void *dbuffer);
int mrvl_ml_init(int argc, char *argv[]);
int mrvl_ml_init_mt(int argc, char *argv[], int num_threads);
int mrvl_ml_model_load(char *model_buffer, int model_size, int num_batches);
int mrvl_ml_model_unload(int model_id);
int mrvl_ml_model_run(struct run_args *run_arg);
int mrvl_ml_model_run_mt(struct run_args *run_arg, int thread_id);
int mrvl_ml_model_finish();

#endif /* __VELOX_MLINTF_H__ */
