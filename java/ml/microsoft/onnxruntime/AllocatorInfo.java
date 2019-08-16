package ml.microsoft.onnxruntime;

public class AllocatorInfo {
  public native static AllocatorInfo createCpu(AllocatorType allocatorType, MemType memType);
}
