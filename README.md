# HLS_CNN

## Introduction
VITIS HLS를 사용한 VGG network의 5개 convolution 구현
## Code explanation
### cnn_single
+ host.cpp
  + 5 convolution x 1 iteration
  + SW와 HW에서 수행 후 각 convolution에 대해 결과 비교
+ cnn.cpp
  + 5개 convolution을 병렬적으로 1회 수행
+ cnn.h
  + 5개 convolution 각각의 dimension, tile size 정의
  + Tm, Tn = 16
### cnn_5_layers
+ host.cpp
  + 5 convolution x 5 iterations
  + Iteration 5의 convolution 5에 대해 sw와 hw 결과 비교
  + Conv1의 결과가 다음 interation의 conv2 input으로 이어짐
  + 각 iteration 후 host에서 max pooling & zero padding 수행
+ cnn.cpp
  + 5개 convolution을 병렬적으로 1회 수행
+ cnn.h
  + 5개 convolution 각각의 dimension, tile size 정의
  + Tm, Tn = 16

## 실행 환경
+ Build : AWS M5 instance
+ Run : AWS F1 instance

## Build
1. M5 instance에서 cnn_single/ or cnn_5_layers/ directory로 이동
```
cd cnn_single/
```
```
cd cnn_5_layers/
```
2. sw_emu
```
make run TARGET=sw_emu DEVICE=$AWS_PLATFORM
```
3. hw_emu
```
make run TARGET=hw_emu DEVICE=$AWS_PLATFORM
```
4. hw (make bitstream)
```
make run TARGET=hw DEVICE=$AWS_PLATFORM
$VITIS_DIR/tools/create_vitis_afi.sh
```
5. Convert xclbin into AFI (AWS S3 bucket 사용)
```
$VITIS_DIR/tools/create_vitis_afi.sh -xclbin =[your synthesis directory]/cnn.xclbin -o=cnn -s3_bucket=[your bucket name] -s3_dcp_key=dcp -s3_logs_key=logs
```
6. Transfer to F1 instance

## Run
1. host, cnn.awsxclbin 을 F1 instance로 copy
2. Setup environment
```
git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
cd $AWS_FPGA_REPO_DIR
source vitis_setup.sh
source vitis_runtime_setup.sh
```
3. Run FPGA bit-stream
```
chmod +x host
./host cnn.awsxclbin
```
