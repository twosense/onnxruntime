package ml.microsoft.onnxruntime;

public class AllocatorInfo {
  private native static AllocatorInfo createCpu(AllocatorType allocatorType, MemType memType);
}
