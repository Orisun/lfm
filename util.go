// Date         : 2020-07-30 13:14
// Author       : zhangzhaoyang
// Description  :

package lfm

import (
	"github.com/ziutek/blas"
	"math"
	"math/rand"
	"sync/atomic"
	"time"
	"unsafe"
)

func Shuffle(arr []*Rating) []*Rating {
	if arr == nil || len(arr) == 0 {
		return nil
	}
	rect := make([]*Rating, len(arr))
	r := rand.New(rand.NewSource(time.Now().Unix()))
	for i, j := range r.Perm(len(arr)) {
		rect[i] = arr[j]
	}
	return rect
}

func ShuffleStringList(arr []string) []string {
	if arr == nil || len(arr) == 0 {
		return nil
	}
	rect := make([]string, len(arr))
	r := rand.New(rand.NewSource(time.Now().Unix()))
	for i, j := range r.Perm(len(arr)) {
		rect[i] = arr[j]
	}
	return rect
}

func AtomicAddFloat32(val *float32, delta float32) (new float32) {
	for {
		old := *val
		new = old + delta
		if atomic.CompareAndSwapUint32(
			(*uint32)(unsafe.Pointer(val)),
			math.Float32bits(old),
			math.Float32bits(new),
		) {
			break
		}
	}
	return
}

func VecDot(a, b []float32, l int) float32 {
	//var sum float32 = 0.0
	//for i := 0; i < l; i++ {
	//	sum += a[i] * b[i]
	//}
	//return sum
	return blas.Sdot(l, a, 1, b, 1) //长度超过16时，用汇编计算向量内积会快一些
}