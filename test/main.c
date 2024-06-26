/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2023 Marvell.
 */

#include <errno.h>
#include <getopt.h>
#include <linux/limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <rte_eal.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <rte_mldev.h>

#include "mldpc.h"

#define EAL_ARGS 2

struct ml_model_files {
	char model_file[PATH_MAX];
	char input_file[PATH_MAX];
	char output_file[PATH_MAX];
};

struct ml_model_files mdl_file;

static int
check_files(void)
{
	/* check file availability */

	if (access(mdl_file.model_file, F_OK) == -1) {
		fprintf(stderr, "Failed %s\n", mdl_file.model_file);
		return -1;
	}

	if (access(mdl_file.input_file, F_OK) == -1) {
		fprintf(stderr, "Failed %s\n", mdl_file.input_file);
		return -1;
	}

	return 0;
}

static void
print_usage(const char *prog_name)
{
	printf("*** Usage: %s [EAL params] --"
	       " -m <model>"
	       " -i <input>"
	       " -o <output>"
	       "\n",
	       prog_name);
	printf("\n");
}

static int
parse_args(int argc, char *argv[])
{
	uint16_t nb_outputs;
	uint16_t nb_inputs;
	uint16_t nb_models;
	int opt_index;
	int opt;

	static struct option longopts[] = {{"model", required_argument, NULL, 'm'},
					   {"input", required_argument, NULL, 'i'},
					   {"output", required_argument, NULL, 'o'},
					   {"help", no_argument, NULL, 'h'},
					   {NULL, 0, NULL, 0}};

	nb_models = 0;
	nb_inputs = 0;
	nb_outputs = 0;

	while ((opt = getopt_long(argc, argv, "m:i:o:h", longopts, &opt_index)) != EOF) {
		switch (opt) {
		case 'm':
			if (nb_models != 0) {
				fprintf(stderr, "Multiple models not supported\n");
				return -1;
			}
			strncpy(mdl_file.model_file, optarg, PATH_MAX - 1);
			nb_models++;
			break;
		case 'i':
			if (nb_inputs != 0) {
				fprintf(stderr, "Multiple inputs not supported\n");
				return -1;
			}
			strncpy(mdl_file.input_file, optarg, PATH_MAX - 1);
			nb_inputs++;
			break;
		case 'o':
			if (nb_outputs != 0) {
				fprintf(stderr, "Multiple outputs not supported\n");
				return -1;
			}
			strncpy(mdl_file.output_file, optarg, PATH_MAX - 1);
			nb_outputs++;
			break;
		case 'h':
			print_usage(argv[0]);
			return 1;
		default:
			print_usage(argv[0]);
			return -1;
		}
	}
	/* check models, inputs and outputs count */
	if (nb_inputs != nb_models) {
		fprintf(stderr,
			"Invalid arguments: "
			"Inputs count (%d) not equal to models count (%d)\n\n",
			nb_inputs, nb_models);
		return -1;
	}

	if (nb_outputs != nb_models) {
		fprintf(stderr,
			"Invalid arguments: "
			"Outputs count (%d) not equal to models count (%d)\n\n",
			nb_outputs, nb_models);
		return -1;
	}

	return check_files();
}

int
main(int argc, char *argv[])
{
	int ret = 0;
	uint64_t model_size;
	uint64_t input_size;
	char *model_addr;
	FILE *fp;
	uint64_t size = 0;

	int model_id;
	void *input_buffer;
	void *output_buffer;
	int num_batches;

	ret = mrvl_ml_init(argc, argv);
	if (ret != 0) {
		printf("Failure in mrvl_ml_init()\n");
		return -1;
	}
	argc -= EAL_ARGS;
	argv += EAL_ARGS;

	ret = parse_args(argc, argv);
	if (ret < 0)
		return ret;
	else if (ret > 0)
		return 0;

	num_batches = 1;

	/* Read the model binary and fill the size in above apis */
	fp = fopen(mdl_file.model_file, "rb");
	fseek(fp, 0, SEEK_END);
	model_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	printf("Model size is %ld\n", model_size);
	model_addr = (char *)malloc(model_size * sizeof(char));
	if (model_addr == NULL) {
		printf("Failed to allocate model buffer\n");
		return -1;
	}

	fread(model_addr, 1, model_size, fp);
	fclose(fp);

	model_id = mrvl_ml_model_load(model_addr, model_size);
	printf("Model id is %d\n", model_id);

	fp = fopen(mdl_file.input_file, "rb");
	fseek(fp, 0, SEEK_END);
	input_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	/* allocate size for dequantize input */
	input_buffer = mrvl_ml_io_alloc(model_id, 1, &size);
	if (input_buffer == NULL) {
		printf("Failed to allocate input buffer\n");
		return -1;
	}
	if (fread(input_buffer, 1, input_size, fp) != input_size) {
		printf("fread failed, err=%s, ecode= %d\n", strerror(errno), errno);
		return -1;
	}
	fclose(fp);

	/* Allocate o/p size */
	output_buffer = mrvl_ml_io_alloc(model_id, 3, &size);
	if (output_buffer == NULL) {
		printf("Failed to allocate output buffer\n");
		return -1;
	}

	ret = mrvl_ml_model_run(model_id, input_buffer, output_buffer, num_batches);
	if (ret != 0) {
		printf("Failure inside mrvl_ml_model_run()\n");
		return -1;
	}

	/* dump output to a file */
	fp = fopen(mdl_file.output_file, "wb");
	fseek(fp, 0, SEEK_SET);
	fwrite(output_buffer, size, 1, fp);
	fclose(fp);

	mrvl_ml_io_free(model_id, 1, input_buffer);
	mrvl_ml_io_free(model_id, 3, output_buffer);

	ret = mrvl_ml_model_unload(model_id);
	if (ret != 0) {
		printf("Failure inside mrvl_ml_model_unload()\n");
		return -1;
	}

	mrvl_ml_finish();

	return 0;
}
