package ml.microsoft.onnxruntime

    import ml.microsoft.onnxruntime.Value

    class Session {
  static {
    System.loadLibrary("onnxruntime_jni");
  }
  public Session() {
    initHandle();
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  private native void initHandle();
  public native void dispose();
  public native Value[] Run(Object runOptions, String[] input_names, Value[] input_values, String[] output_names);
}
