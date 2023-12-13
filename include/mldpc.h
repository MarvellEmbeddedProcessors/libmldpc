/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2023 Marvell.
 */

#ifndef _MLDPC_H_
#define _MLDPC_H_

#include <stdint.h>
#include <stdio.h>

/**
 * ML buffer type
 */
enum mrvl_ml_buffer_type {
	/** Quantized input */
	input_quantize = 0,

	/** Dequantized input */
	input_dequantize,

	/** Quantized output*/
	output_quantize,

	/** Dequantized output */
	output_dequantize
};

/**
 * Initialize ML DataPlane environment.
 *
 * Initializes ML device and internal resources.
 *
 * @param[in] argc
 *   Argument count
 * @param[in] argv
 *   Argument vector
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_init(int argc, char *argv[]);

/**
 * Initialize ML DataPlane environment, with Multi-threading support.
 * Initializes ML device and internal resources.
 * Enables support for multi-threading environment.
 *
 * @param[in] argc
 *   Argument count
 * @param[in] argv
 *   Argument vector
 * @param[in] num_threads
 *   Number of threads
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_init_mt(int argc, char *argv[], int num_threads);

/**
 * Deinitialize ML device and resources.
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_finish(void);

/**
 * Load and start the model into DataPlane space
 *
 * Initialize model context.
 *
 * @param[in] model_buffer
 *   Model buffer pointer
 * @param[in] model_size
 *   Size of the model
 *
 * @return
 *   model_id on success, < 0 on error
 */
int mrvl_ml_model_load(char *model_buffer, int model_size);

/**
 * Stop and unload the ML model.
 *
 * Collect the stats
 *
 * @param[in] model_id
 *   Id of the the model
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_model_unload(int model_id);

/**
 * Allocate the buffer for model in DataPlane space.
 *
 * @param[in] model_id
 *   Id of the the model
 * @param[in] buff_type
 *   Type of the buffer to be allocated
 * @param[in] size
 *   Size of allocated buffer
 *
 * @return
 *   Returns the buffer pointer on success, NULL on error
 */
void *mrvl_ml_io_alloc(int model_id, enum mrvl_ml_buffer_type buff_type, uint64_t *size);

/**
 * Release the model buffer.
 *
 * @param[in] model_id
 *   Id of the the model
 * @param[in] buff_type
 *   Type of the buffer to be release
 * @param[in] addr
 *   Buffer address to free up.
 *
 */
void mrvl_ml_io_free(int model_id, enum mrvl_ml_buffer_type buff_type, void *addr);

/**
 * Quantize model input
 *
 * @param[in] model_id
 *   Id of the the model
 * @param[in] dbuffer
 *   Pointer to input dequantized data buffer
 * @param[in] qbuffer
 *   Pointer to input quantized data buffer
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_model_quantize(int model_id, void *dbuffer, void *qbuffer);

/**
 * Dequantize model output
 *
 * @param[in] model_id
 *   Id of the the model
 * @param[in] qbuffer
 *   Pointer to output quantized data buffer
 * @param[in] dbuffer
 *   Pointer to output dequantized data buffer
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_model_dequantize(int model_id, void *qbuffer, void *dbuffer);

/**
 * Run ML inference operation.
 *
 * @param[in] model_id
 *   Id of the the model
 * @param[in] input_buffer
 *   Pointer to input data buffer
 * @param[in] output_buffer
 *   Pointer to output data buffer
 * @param[in] num_batches
 *   Number of batches
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_model_run(int model_id, void *input_buffer, void *output_buffer, int num_batches);

/**
 * Run ML inference operation using specified thread.
 *
 * @param[in] model_id
 *   Id of the the model
 * @param[in] input_buffer
 *   Pointer to input data buffer
 * @param[in] output_buffer
 *   Pointer to output data buffer
 * @param[in] num_batches
 *   Number of batches
 * @param[in] thread_id
 *   Thread Id to be used for inference run
 *
 * @return
 *   0 on success, < 0 on error
 */
int mrvl_ml_model_run_mt(int model_id, void *input_buffer, void *output_buffer, int num_batches,
			 int thread_id);

#endif /* _MLDPC_H_ */
