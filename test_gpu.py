import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\nFound GPU(s):")
    for gpu in gpus:
        print(f"- {gpu.name}")
        # Enable memory growth
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Verify GPU is being used
    print("\nGPU Device Name:", tf.test.gpu_device_name())
    print("Is GPU available:", tf.test.is_gpu_available())
    
    # Run a simple operation to verify GPU usage
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("\nMatrix multiplication completed on GPU")
else:
    print("\nNo GPU found. Please check your CUDA and cuDNN installation.")
    print("Make sure you have installed:")
    print("1. CUDA Toolkit 11.8")
    print("2. cuDNN 8.6 for CUDA 11.x")
    print("3. Added the correct environment variables")
    print("4. Restarted your computer") 