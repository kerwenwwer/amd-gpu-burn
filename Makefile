ifneq ("$(wildcard /opt/rocm/bin/hipcc)", "")
ROCMPATH ?= /opt/rocm
else
$(error HIPCC not found. Please install ROCm toolkit)
endif


HIPCC       := ${ROCMPATH}/bin/hipcc
CCPATH      ?=

override CFLAGS   ?=
override CFLAGS   += -O3
override CFLAGS   += -Wno-unused-result
override CFLAGS   += -I${ROCMPATH}/include
override CFLAGS   += -std=c++11

override LDFLAGS  ?=
override LDFLAGS  += -L${ROCMPATH}/lib
override LDFLAGS  += -Wl,-rpath=${ROCMPATH}/lib
override LDFLAGS  += -lamdhip64
override LDFLAGS  += -lrocblas
override LDFLAGS  += -lhipblas

COMPUTE      ?= gfx90a
HIP_VERSION ?= 6.2.0
IMAGE_DISTRO ?= ubuntu24.04

override HIPCCFLAGS ?=
override HIPCCFLAGS += -I${ROCMPATH}/include
override HIPCCFLAGS += --offload-arch=${COMPUTE}

IMAGE_NAME ?= gpu-burn 

.PHONY: clean

gpu_burn: gpu_burn-drv.o compare.hsaco
	${HIPCC} -o $@ $< -O3 ${LDFLAGS}

%.o: %.cpp
	${HIPCC} ${CFLAGS} -c $<

%.hsaco: %.hip
	${HIPCC} ${HIPCCFLAGS} --genco $< -o $@

clean:
	$(RM) *.hsaco *.o gpu_burn

image:
	docker build --build-arg HIP_VERSION=${HIP_VERSION} --build-arg IMAGE_DISTRO=${IMAGE_DISTRO} -t ${IMAGE_NAME} .