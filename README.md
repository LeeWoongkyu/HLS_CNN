# HLS_CNN

## Introduction
VITIS HLS를 사용한 VGG network의 5개 convolution 구현
### cnn_single
+ 5개 convolution을 병렬적으로 1회씩 수행
### cnn_5_layers
+ 5개 convolution에 대한 병렬 수행을 5번 반복, conv1에 대한 결과가 다음 conv2의 input으로 이어짐
+ 1 iteration 후 host에서 max pooling & zero padding 적용

## 실행 환경
+ Build : AWS M5 instance
+ Run : AWS F1 instance

## Build
1. cnn_single/ or cnn_5_layers/ directory를 M5 instance로 copy
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
5. Convert xclbin into AFI, (AWS S3 bucket 사용)
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
