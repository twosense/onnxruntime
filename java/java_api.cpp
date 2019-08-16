#include "jni_helper.h"

#include "onnxruntime_cxx_api.h"

using namespace Ort;
extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Env_initHandle(JNIEnv* env, jobject obj /* this */) {
  // Ort::Env* ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
  OrtEnv* ort_env;
  ORT_THROW_ON_ERROR(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &ort_env));
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
  // auto* session = new Session(ort_env, JStringtoStdString(env, j_model_path).c_str(), options);
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

  jclass cls = env->FindClass("ml/microsoft/onnxruntime/Value");
  auto j_value_arr = env->NewObjectArray(output_count, cls, nullptr);
  for (int i = 0; i < output_count; i++) {
    jobject j_value = env->AllocObject(cls);
    setHandle(env, j_value, output_values[i]);
    env->SetObjectArrayElement(j_value_arr, i, j_value);
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
Java_ml_microsoft_onnxruntime_Value_createFloatTensorByData(JNIEnv* env, jobject obj /* this */,
                                                            jobject j_allocator_info, 
                                                            jfloatArray j_data, jlongArray j_shape) {
  auto* allocator_info = getHandle<OrtAllocatorInfo>(env, j_allocator_info);
  const auto data_ptr = env->GetFloatArrayElements(j_data, nullptr);
  const auto data_len = env->GetArrayLength(j_data);
  const auto shape_ptr = env->GetLongArrayElements(j_shape, nullptr);
  const auto shape_len = env->GetArrayLength(j_shape);
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(allocator_info, data_ptr, data_len, 
              shape_ptr, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out));
  jclass cls = env->FindClass("ml/microsoft/onnxruntime/Value");
  jobject j_value = env->AllocObject(cls);
  setHandle(env, j_value, out);
  return j_value;
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_AllocatorInfo_createCpu(JNIEnv* env, jobject obj /* this */,
                                                      jobject j_allocator_type, jobject j_mem_type) {
  jint allocator_type_value = env->CallIntMethod(j_allocator_type,
                                                 env->GetMethodID(
                                                     env->FindClass("ml/microsoft/onnxruntime/AllocatorInfo"), 
                                                     "ordinal", "()I"));
  jint mem_type_value = env->CallIntMethod(j_mem_type,
                                           env->GetMethodID(
                                               env->FindClass("ml/microsoft/onnxruntime/MemType"), 
                                               "ordinal", "()I"));
  OrtAllocatorInfo* allocator_info;

  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(static_cast<OrtAllocatorType>(allocator_type_value), 
              static_cast<OrtMemType>(mem_type_value), &allocator_info));
  jclass cls = env->FindClass("ml/microsoft/onnxruntime/AllocatorInfo");
  jobject j_value = env->AllocObject(cls);
  setHandle(env, j_value, allocator_info);
  return j_value;
}
