#include "jni_helper.h"

#include "onnxruntime_cxx_api.h"
#include <core/providers/nnapi/nnapi_provider_factory.h>

#include <android/log.h>

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Env_initHandle(JNIEnv* env, jobject obj /* this */, jobject j_logging_level, jstring logid) {
  OrtEnv* ort_env;
  const auto logging_level_value = enumToInt(env, j_logging_level, "ml/microsoft/onnxruntime/LoggingLevel");
  ORT_THROW_ON_ERROR(OrtCreateEnv(static_cast<OrtLoggingLevel>(logging_level_value),
                                  JStringtoStdString(env, logid).c_str(), &ort_env));
  setHandle(env, obj, ort_env);
}

using namespace Ort;
extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_RunOptions_initHandle(JNIEnv* env, jobject obj /* this */) {
  OrtRunOptions* run_options;
  ORT_THROW_ON_ERROR(OrtCreateRunOptions(&run_options));

  setHandle(env, obj, run_options);
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Session_initHandle(JNIEnv* env, jobject obj /* this */,
                                                 jobject j_ort_env, jstring j_model_path, 
                                                 const jobject j_options) {
  auto* ort_env = getHandle<OrtEnv>(env, j_ort_env);
  const auto* options = getHandle<OrtSessionOptions>(env, j_options);
  OrtSession* session;
  ORT_THROW_ON_ERROR(OrtCreateSession(ort_env, 
              JStringtoStdString(env, j_model_path).c_str(), options, &session));
  setHandle(env, obj, session);
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_initHandle(JNIEnv* env, jobject obj /* this */) {
  OrtSessionOptions* session_options;
  ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_options));

  setHandle(env, obj, session_options);
}

#define DEFINE_DISPOSE(class_name)                                                                \
  extern "C" JNIEXPORT void JNICALL                                                               \
      Java_ml_microsoft_onnxruntime_##class_name##_dispose(JNIEnv* env, jobject obj /* this */) { \
    auto handle = getHandle<Ort##class_name>(env, obj);                                           \
    OrtRelease##class_name(handle);                                                               \
    handle = nullptr;                                                                             \
  }

DEFINE_DISPOSE(SessionOptions);
DEFINE_DISPOSE(Session);
DEFINE_DISPOSE(RunOptions);
DEFINE_DISPOSE(Env);
DEFINE_DISPOSE(Value);

#undef DEFINE_DISPOSE

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_setThreadPoolSize(JNIEnv* env, jobject obj /* this */,
        jint session_thread_pool_size) {
    auto *session_options = getHandle<OrtSessionOptions>(env, obj);
    ORT_THROW_ON_ERROR(OrtSetSessionThreadPoolSize(session_options, session_thread_pool_size));
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_setGraphOptimizationLevel(JNIEnv* env, jobject obj /* this */,
        jint graph_optimization_level) {
    auto *session_options = getHandle<OrtSessionOptions>(env, obj);
    ORT_THROW_ON_ERROR(OrtSetSessionGraphOptimizationLevel(session_options, graph_optimization_level));
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_enableProfiling(JNIEnv* env, jobject obj /* this */) {
    auto *session_options = getHandle<OrtSessionOptions>(env, obj);
    ORT_THROW_ON_ERROR(OrtEnableProfiling(session_options, ""));
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_disableProfiling(JNIEnv* env, jobject obj /* this */) {
    auto *session_options = getHandle<OrtSessionOptions>(env, obj);
    ORT_THROW_ON_ERROR(OrtDisableProfiling(session_options));
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_enableCpuMemArena(JNIEnv* env, jobject obj /* this */) {
    auto *session_options = getHandle<OrtSessionOptions>(env, obj);
    ORT_THROW_ON_ERROR(OrtEnableCpuMemArena(session_options));
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_disableCpuMemArena(JNIEnv* env, jobject obj /* this */) {
    auto *session_options = getHandle<OrtSessionOptions>(env, obj);
    ORT_THROW_ON_ERROR(OrtDisableCpuMemArena(session_options));
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_appendNnapiExecutionProvider(JNIEnv* env, jobject obj /* this */) {
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options));
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ml_microsoft_onnxruntime_Session_run(JNIEnv* env, jobject obj /* this */,
                                          jobject j_run_options, jobjectArray j_input_names, 
                                          jobjectArray j_input_values, jobjectArray j_output_names) {
  auto* session = getHandle<OrtSession>(env, obj);
  auto* run_options = getHandle<OrtRunOptions>(env, j_run_options);
  auto input_count = env->GetArrayLength(j_input_values);
  auto output_count = env->GetArrayLength(j_output_names);
  std::vector<const char*> input_names;
  std::vector<OrtValue*> input_values;
  for (int i = 0; i < input_count; i++) {
    auto jstr = env->GetObjectArrayElement(j_input_names, i);
    const char* char_ptr = env->GetStringUTFChars(static_cast<jstring>(jstr), nullptr);
    input_names.push_back(char_ptr);

    auto j_value = env->GetObjectArrayElement(j_input_values, i);
    auto value_ptr = getHandle<OrtValue>(env, j_value);
    input_values.push_back(value_ptr);
  }
  std::vector<const char*> output_names;
  for (int i = 0; i < output_count; i++) {
    auto jstr = env->GetObjectArrayElement(j_output_names, i);
    const char* char_ptr = env->GetStringUTFChars(static_cast<jstring>(jstr), nullptr);
    output_names.push_back(char_ptr);
  }

  std::vector<OrtValue*> output_values(output_count, nullptr);
  ORT_THROW_ON_ERROR(OrtRun(session, run_options, input_names.data(), input_values.data(), 
              input_count, output_names.data(), output_count, output_values.data()));

  const char* class_name = "ml/microsoft/onnxruntime/Value";
  jclass cls = env->FindClass(class_name);
  auto j_value_arr = env->NewObjectArray(output_count, cls, nullptr);
  for (int i = 0; i < output_count; i++) {
    env->SetObjectArrayElement(j_value_arr, i, newObject(env, class_name, output_values[i]));
  }
  return j_value_arr;
}

// extern "C" JNIEXPORT void JNICALL
// Java_ml_microsoft_onnxruntime_Allocator_initHandle(JNIEnv *env, jobject obj /* this */) {
//     auto *allocator = Allocator::CreateDefault();
//     setHandle(env, obj, session);
// }
//

// extern "C" JNIEXPORT jobject JNICALL
// Java_ml_microsoft_onnxruntime_Value_CreateFloatTensor(JNIEnv* env, jobject obj /* this */,
//                                                       jobject j_allocator, jlongArray j_shape) {
//   auto* allocator = getHandle<Allocator>(env, j_allocator);
//   const auto shape_ptr = env->GetLongArrayElements(j_shape, nullptr);
//   const auto shape_len = env->GetArrayLength(j_shape);
//   auto value = Value::CreateTensor<float>(*allocator, shape_ptr, shape_len);
//   jclass cls = env->FindClass("ml/microsoft/onnxruntime/Value");
//   jobject j_value = env->AllocObject(cls);
//   setHandle(env, j_value, &value);
// }
//
extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_createFloatTensorWithData(JNIEnv* env, jobject /* this */,
                                                              jobject j_allocator_info,
                                                              jfloatArray j_data, jlongArray j_shape) {
  auto* allocator_info = getHandle<OrtAllocatorInfo>(env, j_allocator_info);
  const auto data_ptr = env->GetFloatArrayElements(j_data, nullptr);
  const auto data_len = env->GetArrayLength(j_data);
  const auto shape_ptr = env->GetLongArrayElements(j_shape, nullptr);
  const auto shape_len = env->GetArrayLength(j_shape);
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(allocator_info, data_ptr, data_len * sizeof(float),
                                                       shape_ptr, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out));
  return newObject(env, "ml/microsoft/onnxruntime/Value", out);
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_getTensorMutableData(JNIEnv* env, jobject obj /* this */) {
  auto* value = getHandle<OrtValue>(env, obj);
  uint8_t* out;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(value, (void**)&out));
  size_t count;
  OrtTensorTypeAndShapeInfo* info;
  ORT_THROW_ON_ERROR(OrtGetTensorTypeAndShape(value, &info));
  ORT_THROW_ON_ERROR(OrtGetTensorShapeElementCount(info, &count));
  auto byte_buf = env->NewDirectByteBuffer(out, count * 4);
  return byte_buf;
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_getTensorTypeAndShapeInfo(JNIEnv* env, jobject obj /* this */) {
  auto* value = getHandle<OrtValue>(env, obj);
  OrtTensorTypeAndShapeInfo* info;
  ORT_THROW_ON_ERROR(OrtGetTensorTypeAndShape(value, &info));

  return newObject(env, "ml/microsoft/onnxruntime/TensorTypeAndShapeInfo", info);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ml_microsoft_onnxruntime_TensorTypeAndShapeInfo_getElementCount(JNIEnv* env, jobject obj /* this */) {
  auto* info = getHandle<OrtTensorTypeAndShapeInfo>(env, obj);

  size_t count;
  ORT_THROW_ON_ERROR(OrtGetTensorShapeElementCount(info, &count));

  return static_cast<jlong>(count);
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_AllocatorInfo_createCpu(JNIEnv* env, jobject /* this */,
                                                      jobject j_allocator_type, jobject j_mem_type) {
  jint allocator_type_value = enumToInt(env, j_allocator_type, "ml/microsoft/onnxruntime/AllocatorType");
  jint mem_type_value = enumToInt(env, j_mem_type, "ml/microsoft/onnxruntime/MemType");
  OrtAllocatorInfo* allocator_info;

  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(static_cast<OrtAllocatorType>(allocator_type_value), 
              static_cast<OrtMemType>(mem_type_value), &allocator_info));
  return newObject(env, "ml/microsoft/onnxruntime/AllocatorInfo", allocator_info);
}
