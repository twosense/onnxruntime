// Copyright 2019 JD.com Inc. JD AI

#ifndef _HANDLE_H_INCLUDED_
#define _HANDLE_H_INCLUDED_

#include <jni.h>

#include <string>

inline jfieldID getHandleField(JNIEnv* env, jobject obj) {
  jclass c = env->GetObjectClass(obj);
  // J is the type signature for long:
  return env->GetFieldID(c, "nativeHandle", "J");
}

template <typename T>
T* getHandle(JNIEnv* env, jobject obj) {
  jlong handle = env->GetLongField(obj, getHandleField(env, obj));
  return reinterpret_cast<T*>(handle);
}

inline void setHandle(JNIEnv* env, jobject obj, void* t) {
  jlong handle = reinterpret_cast<jlong>(t);
  env->SetLongField(obj, getHandleField(env, obj), handle);
}

inline std::string JStringtoStdString(JNIEnv* env, jstring j_str) {
  const char* native_char_ptr = env->GetStringUTFChars(j_str, nullptr);
  std::string native_str(native_char_ptr);
  env->ReleaseStringUTFChars(j_str, native_char_ptr);
  return native_str;
}

#endif
