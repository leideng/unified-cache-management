/**
 * Stub for CANN base/err_msg.h when the toolkit does not provide it.
 * Provides REPORT_INNER_ERR_MSG as a no-op so that npu_ops can build.
 */
#ifndef BASE_ERR_MSG_H
#define BASE_ERR_MSG_H

/* No-op stub: real CANN would report the error via error_manager */
#define REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ...) do { (void)(ERR_CODE_STR); } while (0)

#endif /* BASE_ERR_MSG_H */
