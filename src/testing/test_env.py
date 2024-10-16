import sys


def check_python_version():
    print(f"Python version: {sys.version}")


def check_pytorch():
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f" - Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Using CPU.")
    except ImportError:
        print("PyTorch is not installed.")


def main():
    print("Starting environment tests...\n")
    check_python_version()
    print()
    check_pytorch()
    print()
    print("\nEnvironment tests completed.")


if __name__ == "__main__":
    main()
