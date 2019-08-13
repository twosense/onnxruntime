#include "jni_helper.h"

#include "onnxruntime_cxx_api.h"

using namespace Ort;
extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Env_initHandle(JNIEnv* env, jobject obj /* this */) {
  Ort::Env* ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
  setHandle(env, obj, ort_env);
}

using namespace Ort;
extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_RunOptions_initHandle(JNIEnv* env, jobject obj /* this */) {
  auto* run_options = new RunOptions();
  setHandle(env, obj, run_options);
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Session_initHandle(JNIEnv* env, jobject obj /* this */,
                                                 jobject j_ort_env, jstring j_model_path, const jobject j_options) {
  auto* ort_env = getHandle<Env>(env, j_ort_env);
  const auto* options = getHandle<SessionOptions>(env, j_options);
  auto* session = new Session(*ort_env, "", *options);
  setHandle(env, obj, session);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ml_microsoft_onnxruntime_Session_Run(JNIEnv* env, jobject obj /* this */,
                                          jobject j_run_options, jobjectArray j_input_names, jobjectArray j_input_values, jobjectArray j_output_names) {
  auto* session = getHandle<Session>(env, obj);
  auto* run_options = getHandle<RunOptions>(env, j_run_options);
  auto input_count = env->GetArrayLength(j_input_values);
  auto output_count = env->GetArrayLength(j_output_names);
  std::vector<const char*> input_names;
  std::vector<Value> input_values;
  for (int i = 0; i < input_count; i++) {
    auto jstr = env->GetObjectArrayElement(j_input_names, i);
    const char* char_ptr = env->GetStringUTFChars(static_cast<jstring>(jstr), nullptr);
    input_names.push_back(char_ptr);

    auto j_value = env->GetObjectArrayElement(j_input_values, i);
    auto value_ptr = getHandle<Value>(env, j_value);
    input_values.push_back(std::move(*value_ptr));
  }
  std::vector<const char*> output_names;
  for (int i = 0; i < output_count; i++) {
    auto jstr = env->GetObjectArrayElement(j_output_names, i);
    const char* char_ptr = env->GetStringUTFChars(static_cast<jstring>(jstr), nullptr);
    output_names.push_back(char_ptr);
  }

  auto output_values = session->Run(*run_options, input_names.data(), input_values.data(), input_count, output_names.data(), output_count);

  // Restore input_values
  for (int i = 0; i < input_count; i++) {
    auto j_value = env->GetObjectArrayElement(j_input_values, i);
    setHandle(env, j_value, std::move(input_values[i]));
  }

  jclass cls = env->FindClass("ml/microsoft/onnxruntime/Value");
  auto j_value_arr = env->NewObjectArray(output_count, cls, nullptr);
  for (int i = 0; i < output_count; i++) {
    jobject j_value = env->AllocObject(cls);
    setHandle(env, j_value, &output_values[i]);
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

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_CreateFloatTensor(JNIEnv* env, jobject obj /* this */,
                                                      jobject j_allocator, jlongArray j_shape) {
  auto* allocator = getHandle<Allocator>(env, j_allocator);
  const auto shape_ptr = env->GetLongArrayElements(j_shape, nullptr);
  const auto shape_len = env->GetArrayLength(j_shape);
  auto value = Value::CreateTensor<float>(*allocator, shape_ptr, shape_len);
}
