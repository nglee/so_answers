https://stackoverflow.com/a/44678423/7724939

`nvcc vecAdd.cu timer.c -o vecAdd` fails if `extern "C" {}` surrounding `#include "timer.h"` is removed in `vecAdd.cu`.
