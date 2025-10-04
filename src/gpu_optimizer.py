# src/gpu_optimizer.py
import tensorflow as tf


def optimize_gpu_for_1650():
    """Optimize GPU settings for GTX 1650 (4GB VRAM)"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Method 1: Memory growth (prevents TF from taking all memory at once)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Method 2: Set memory limit (3.5GB for stability)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]
            )

            print("✓ GPU optimized for GTX 1650 (3.5GB limit + memory growth)")
            return True

        except RuntimeError as e:
            print(f"GPU optimization failed: {e}")
            print("Trying alternative configuration...")

            # Alternative: Just enable memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✓ GPU memory growth enabled")
                return True
            except:
                print("✗ Could not optimize GPU")
                return False
    else:
        print("No GPU found - using CPU")
        return False


if __name__ == "__main__":
    optimize_gpu_for_1650()
