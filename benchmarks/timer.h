#pragma once

// Wrapper for CUDA GPU timers
struct Timer {
	cudaEvent_t before, after;
	float time;

	inline void start() {
		cudaEventRecord(before);
	}

	inline float stop() {
		cudaEventRecord(after);
		cudaEventSynchronize(after);
		cudaEventElapsedTime(&time, before, after);
		return time;
	}

	Timer() {
		cudaEventCreate(&before);
		cudaEventCreate(&after);
	}

	~Timer() {
		cudaEventDestroy(before);
		cudaEventDestroy(after);
	}
};
