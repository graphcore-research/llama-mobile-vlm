# ACL benchmark

**Installing**

```sh
wget https://github.com/ARM-software/ComputeLibrary/releases/download/v52.6.0/arm_compute-v52.6.0-linux-aarch64-cpu-bin.tar.gz -O acl.tar.gz
mkdir -p acl
tar -xzf acl.tar.gz --strip-components=1 -C acl
```

**Running (our benchmark)**

```sh
mkdir -p bin
clang++ -stdlib=libc++ -isystem acl -isystem acl/include -O3 -Wall -Werror -fopenmp bench_acl.cpp -o bin/bench_acl -L acl/lib/armv8a-neon/ -larm_compute && env LD_LIBRARY_PATH=acl/lib/armv8a-neon ./bin/bench_acl
```

**Running (their example)**

```sh
mkdir -p bin
clang++ -stdlib=libc++ -isystem acl -isystem acl/include -O3 acl/examples/neon_matmul.cpp acl/utils/Utils.cpp -o bin/neon_matmul -L acl/lib/armv8a-neon/ -larm_compute
env LD_LIBRARY_PATH=acl/lib/armv8a-neon ./bin/neon_matmul
```
