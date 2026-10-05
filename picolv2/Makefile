TARGET ?= pico
BUNDLE_ROOT ?= $(CURDIR)/build/picolv2/$(TARGET)
BUNDLE_DIR := $(BUNDLE_ROOT)/neuralseed.lv2
SDK_ROOT ?= ../../../../picolv2/plugin-src/sdk
SDK_RUNTIME ?= $(SDK_ROOT)/src/runtime.c
SDK_INCLUDE ?= $(SDK_ROOT)/include
COMMON := -O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -DRTNEURAL_DEFAULT_ALIGNMENT=8 -DRTNEURAL_NO_DEBUG=1 -I. -I../RTNeural -I../DaisySP/Source -I../DaisySP/Source/Utility -I$(SDK_INCLUDE)
SOURCES := neuralseed_lv2.cpp model_json.cpp ../DaisySP/Source/Filters/svf.cpp

ifeq ($(TARGET),pico)
CXX := arm-none-eabi-g++
CC := arm-none-eabi-gcc
CXXFLAGS := $(COMMON) -std=c++17 -mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 -fPIC -fno-builtin -DPICOLV2_TARGET -DNEURALSEED_FAST_ACTIVATIONS
CFLAGS := -O3 -mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 -fPIC -ffreestanding -fno-builtin -I$(SDK_INCLUDE)
LDFLAGS := -mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 -shared -nostdlib -Wl,-Bsymbolic -Wl,-z,undefs -Wl,--wrap=tanhf -Wl,--gc-sections -Wl,--version-script=exports.map -Wl,-z,noexecstack -Wl,-z,max-page-size=0x1000 -Wl,--no-warnings
LDLIBS := -Wl,--start-group -lstdc++ -lsupc++ -lc -lm -lnosys -lgcc -Wl,--end-group
RUNTIME := build/pico/runtime.o
FAST_OBJECT := build/pico/fast_math.o
else ifeq ($(TARGET),pc)
CXX := g++
CXXFLAGS := $(COMMON) -std=c++17 -fPIC
LDFLAGS := -shared -Wl,-Bsymbolic -Wl,--no-undefined -Wl,--gc-sections -Wl,--version-script=exports.map -Wl,-z,noexecstack
LDLIBS := -lm
RUNTIME :=
FAST_OBJECT :=
else ifeq ($(TARGET),pcfast)
CXX := g++
CXXFLAGS := $(COMMON) -std=c++17 -fPIC -DNEURALSEED_FAST_ACTIVATIONS
LDFLAGS := -shared -Wl,-Bsymbolic -Wl,--no-undefined -Wl,--wrap=tanhf -Wl,--gc-sections -Wl,--version-script=exports.map -Wl,-z,noexecstack
LDLIBS := -lm
RUNTIME :=
FAST_OBJECT := build/pcfast/fast_math.o
else ifeq ($(TARGET),pcdynamic)
CXX := g++
CXXFLAGS := $(COMMON) -std=c++17 -fPIC -DNEURALSEED_FAST_ACTIVATIONS -DNEURALSEED_FORCE_DYNAMIC
LDFLAGS := -shared -Wl,-Bsymbolic -Wl,--no-undefined -Wl,--wrap=tanhf -Wl,--gc-sections -Wl,--version-script=exports.map -Wl,-z,noexecstack
LDLIBS := -lm
RUNTIME :=
FAST_OBJECT := build/pcdynamic/fast_math.o
else
$(error Unknown TARGET $(TARGET))
endif

OBJECTS := $(patsubst %.cpp,build/$(TARGET)/%.o,$(notdir $(SOURCES)))
OUT := build/$(TARGET)/plugin.so

.PHONY: all bundle test test-fast
all: $(OUT)
build/$(TARGET)/neuralseed_lv2.o: neuralseed_lv2.cpp model_json.h lv2_abi.h fast_math.h
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/$(TARGET)/model_json.o: model_json.cpp model_json.h
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/$(TARGET)/svf.o: ../DaisySP/Source/Filters/svf.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/$(TARGET)/fast_math.o: fast_math.cpp fast_math.h fast_tanh_table.h
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/pico/runtime.o: $(SDK_RUNTIME)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT): $(OBJECTS) $(FAST_OBJECT) $(RUNTIME) exports.map
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(FAST_OBJECT) $(RUNTIME) $(LDLIBS)
ifeq ($(TARGET),pico)
	arm-none-eabi-strip --strip-unneeded $@
	@arm-none-eabi-size -B $@ | awk 'NR == 2 && $$4 > 131072 { print "Pico image exceeds 128 KiB: " $$4; exit 1 }'
endif
bundle: $(OUT)
	mkdir -p $(BUNDLE_DIR)
	cp neuralseed.lv2/*.ttl neuralseed.lv2/picolv2-resources.txt neuralseed.lv2/model.json $(BUNDLE_DIR)/
	cp $(OUT) $(BUNDLE_DIR)/plugin.so
test:
	mkdir -p build
	g++ -std=c++17 -O2 -I. test_model_json.cpp model_json.cpp -o build/model_json_test
	build/model_json_test
	$(MAKE) TARGET=pc all
	g++ -std=c++17 -O2 -I. test_lv2.cpp -ldl -o build/lv2_test
	build/lv2_test
test-fast:
	$(MAKE) TARGET=pc bundle
	$(MAKE) TARGET=pcfast bundle
	$(MAKE) TARGET=pcdynamic bundle
	g++ -std=c++17 -O2 -I. test_fast_math.cpp fast_math.cpp -ldl -o build/fast_math_test
	build/fast_math_test
