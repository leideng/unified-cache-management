/**
 * Stub for CANN base/alog_pub.h when the toolkit does not provide it
 * (e.g. minimal or alternate CANN layout). Provides minimal declarations
 * so that npu_ops can configure and build.
 */
#ifndef BASE_ALOG_PUB_H
#define BASE_ALOG_PUB_H

#ifdef __cplusplus
extern "C" {
#endif

/* Log level constants */
#define DLOG_DEBUG   0
#define DLOG_INFO   1
#define DLOG_WARN   2
#define DLOG_ERROR  3
#define DLOG_EVENT  4

#define DLOG_TYPE_DEBUG 0

/* Max message length used by OPS_LOG_STUB_FULL */
#ifndef MSG_LENGTH
#define MSG_LENGTH 16384
#endif

/**
 * Check if the given log level is enabled for the module.
 * @return 1 if enabled, 0 otherwise
 */
int AlogCheckDebugLevel(int mod_id, int level);

/**
 * Record a log message.
 */
void AlogRecord(int mod_id, int type, int level, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* BASE_ALOG_PUB_H */
