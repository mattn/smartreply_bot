#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
//#include "smartreply/model.h"
//#include "smartreply/op_resolver.h"
namespace tflite {
namespace ops {
namespace builtin {
// Forward-declarations for the builtin ops.
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_LSH_PROJECTION();
TfLiteRegistration* Register_HASHTABLE_LOOKUP();
}  // namespace builtin
namespace custom {
// Forward-declarations for the custom ops.
TfLiteRegistration* Register_NORMALIZE();
TfLiteRegistration* Register_EXTRACT_FEATURES();
TfLiteRegistration* Register_PREDICT();
}  // namespace custom
}  // namespace ops
}  // namespace tflite
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {
  resolver->AddBuiltin(::tflite::BuiltinOperator_SKIP_GRAM, ::tflite::ops::builtin::Register_SKIP_GRAM());
  resolver->AddBuiltin(::tflite::BuiltinOperator_LSH_PROJECTION, ::tflite::ops::builtin::Register_LSH_PROJECTION());
  resolver->AddBuiltin(::tflite::BuiltinOperator_HASHTABLE_LOOKUP, ::tflite::ops::builtin::Register_HASHTABLE_LOOKUP());
  resolver->AddCustom("Normalize", ::tflite::ops::custom::Register_NORMALIZE());
  resolver->AddCustom("ExtractFeatures", ::tflite::ops::custom::Register_EXTRACT_FEATURES());
  resolver->AddCustom("Predict", ::tflite::ops::custom::Register_PREDICT());
}
