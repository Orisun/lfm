// Date         : 2020-07-27 13:16
// Author       : zhangzhaoyang
// Description  : 参考 https://www.cnblogs.com/zhangchaoyang/articles/5517186.html

package lfm

import (
	"bufio"
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/golang/glog"
	"github.com/ziutek/blas"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

var (
	parallel = flag.Int("parallel", 10, "traning parallelism")
	decay    = flag.Float64("time_decay", 0.0, "time decay exponentiation") //该值越大，时间衰减得越狠。为0时不做时间衰减，此时对训练文件后缀不做要求
	corpus   chan *Rating
)

type Rating struct {
	Uid    int64
	ItemId int64
	Rate   float32
}

type SVD struct {
	//注意：使用gob序列化保存模型时，只会保存以大写字母开头的成员变量

	//模型参数。训练过程中会并发读写这些slice
	F  int
	P  [][]float32
	Q  [][]float32
	Mu float32 //𝜇表示训练集中的所有评分的平均值，𝜇直接由训练集统计得到
	Bu []float32
	Bi []float32

	//把id映射为从0开始递增的序号，因为模型参数用的是数组不是map
	UidIndex    map[int64]int
	ItemIdIndex map[int64]int

	//训练超参
	learningRate float32
	lambda       float32 //正则项系数
	epochs       int
	trainFiles   []string

	//记录训练误差
	error  float32
	sample uint32
}

func (self *SVD) clearError() {
	self.error = 0.0
	self.sample = 0
}

func (self *SVD) addError(err float32) {
	AtomicAddFloat32(&self.error, err*err) //误差的平方
	atomic.AddUint32(&self.sample, 1)
}

func (self *SVD) getError() float32 {
	return self.error / float32(self.sample)
}

//InitParam 如果要对rate执行时间衰减，则trainFile的文件名必须以yyyymmdd结尾。trainFile里每行的格式uid itemid:rate itemid:rate ......
func (self *SVD) InitParam(F int, LearningRate, Lambda float32, Epochs int, trainFiles []string) {
	self.F = F
	self.learningRate = LearningRate
	self.lambda = Lambda
	self.epochs = Epochs
	self.trainFiles = trainFiles
	self.statisticCorpus()
	UserCount := len(self.UidIndex)
	ItemCount := len(self.ItemIdIndex)
	glog.Infof("uid count %d, itemid count %d", UserCount, ItemCount)
	self.P = make([][]float32, UserCount)
	self.Bu = make([]float32, UserCount)
	for i := 0; i < UserCount; i++ {
		self.Bu[i] = 0
		arr := make([]float32, F)
		for j := 0; j < F; j++ {
			arr[j] = rand.Float32() / float32(math.Sqrt(float64(F))) //确保初始化之后内积小于1
		}
		self.P[i] = arr
	}
	self.Q = make([][]float32, ItemCount)
	self.Bi = make([]float32, ItemCount)
	for i := 0; i < ItemCount; i++ {
		self.Bi[i] = 0
		arr := make([]float32, F)
		for j := 0; j < F; j++ {
			arr[j] = rand.Float32() / float32(math.Sqrt(float64(F)))
		}
		self.Q[i] = arr
	}
	self.clearError()
	glog.Infof("init param finish")
}

func (self *SVD) timeDecay(ago time.Time) float64 {
	interval := math.Floor(time.Now().Sub(ago).Hours() / 24) //精确到天
	return math.Exp(-*decay * interval)
}

func (self *SVD) statisticCorpus() error {
	var rateSum float32
	var rateCount int32
	uidMap := make(map[int64]bool)
	itemIdMap := make(map[int64]bool)

	for _, infile := range self.trainFiles {
		decayCoef := 1.0
		if *decay > 0 {
			timeloc, _ := time.LoadLocation("Asia/Shanghai")
			if len(infile) > 8 {
				postfix := infile[len(infile)-8:]
				if day, err := time.ParseInLocation("20060102", postfix, timeloc); err != nil {
					return err
				} else {
					decayCoef = self.timeDecay(day)
				}
			} else {
				return fmt.Errorf("train file name must end with date if you need time decay for rate")
			}
		}

		file, err := os.OpenFile(infile, os.O_RDONLY, os.ModePerm)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					return err
				} else {
					break
				}
			}
			arr := strings.Split(strings.Trim(line, "\n"), " ")
			if len(arr) >= 2 {
				if uid, err := strconv.ParseInt(arr[0], 10, 64); err != nil {
					glog.Errorf("parse uid failed: %s", arr[0])
					continue
				} else {
					uidMap[uid] = true
					for _, ele := range arr[1:] {
						brr := strings.Split(ele, ":")
						if len(brr) == 2 {
							if itemid, err := strconv.ParseInt(brr[0], 10, 64); err != nil {
								glog.Errorf("parse itemid failed: %s", ele)
								continue
							} else {
								itemIdMap[itemid] = true
								if rate, err := strconv.ParseFloat(brr[1], 64); err != nil {
									glog.Errorf("parse rate failed: %s", brr[1])
									continue
								} else {
									rateSum += float32(rate * decayCoef)
									rateCount++
								}
							}
						}
					}
				}
			}
		}
	}

	self.Mu = rateSum / float32(rateCount)
	glog.Infof("total %d rate, average is %f", rateCount, self.Mu)
	self.UidIndex = make(map[int64]int)
	index := 0
	for uid, _ := range uidMap {
		self.UidIndex[uid] = index
		index++
	}
	self.ItemIdIndex = make(map[int64]int)
	index = 0
	for itemid, _ := range itemIdMap {
		self.ItemIdIndex[itemid] = index
		index++
	}
	return nil
}

func (self *SVD) Predict(uid, itemid int64) (score float32, userIndex, itemIndex int) {
	userIndex = -1
	itemIndex = -1
	var ok bool
	if userIndex, ok = self.UidIndex[uid]; ok {
		if itemIndex, ok = self.ItemIdIndex[itemid]; ok {
			dot := self.vecDot(self.P[userIndex], self.Q[itemIndex], self.F)
			score = dot + self.Mu + self.Bu[userIndex] + self.Bi[itemIndex]
			if math.IsNaN(float64(score)) {
				glog.Errorf("score %f dot %f Mu %f Bu %f Bi %f", score, dot, self.Mu, self.Bu[userIndex], self.Bi[itemIndex])
				syscall.Exit(1)
			}
		} else {
			glog.Errorf("could not found index of itemid %d", itemid)
		}
	} else {
		glog.Errorf("could not found index of uid %d", uid)
	}
	return
}

func (self *SVD) vecDot(a, b []float32, l int) float32 {
	//var sum float32 = 0.0
	//for i := 0; i < l; i++ {
	//	sum += a[i] * b[i]
	//}
	//return sum
	return blas.Sdot(l, a, 1, b, 1) //长度超过16时，用汇编计算向量内积会快一些
}

func (self *SVD) Train() {
	for iter := 0; iter < self.epochs; iter++ {
		self.train()
		self.learningRate *= 0.9 //每轮迭代后，学习率要衰减
		glog.Infof("iteration %d train finish, learning rate is %f, mse is %f", iter, self.learningRate, self.getError())
	}
	glog.Infof("train over")
}

func (self *SVD) train() {
	corpus = make(chan *Rating, 10000)
	stopSignal := make(chan bool, *parallel)
	wg := sync.WaitGroup{}
	wg.Add(*parallel)
	for i := 0; i < *parallel; i++ {
		go func() {
			defer wg.Done()
		LOOP:
			for {
				select {
				case <-stopSignal:
					break LOOP //注意：只写一个break不会跳出for循环！！！
				case rating := <-corpus:
					self.update(rating)
				}
			}
		}()
	}

	self.trainFiles = ShuffleStringList(self.trainFiles) //对训练文件进行shuffle
	for _, trainFile := range self.trainFiles {
		if err := self.parseTrainFile(trainFile, corpus); err != nil {
			glog.Errorf("read train file %s failed %v", trainFile, err)
		}
		glog.Infof("read train file %s finish", trainFile)
	}
	for i := 0; i < *parallel; i++ {
		stopSignal <- true
	}
	wg.Wait()
	close(corpus)
	for rating := range corpus {
		self.update(rating)
	}
}

func (self *SVD) parseTrainFile(infile string, channel chan *Rating) error {
	decayCoef := 1.0
	if *decay > 0 {
		if len(infile) > 8 {
			postfix := infile[len(infile)-8:]
			timeloc, _ := time.LoadLocation("Asia/Shanghai")
			if day, err := time.ParseInLocation("20060102", postfix, timeloc); err != nil {
				return err
			} else {
				decayCoef = self.timeDecay(day)
			}

		} else {
			return fmt.Errorf("train file name must end with date if you need time decay for rate")
		}
	}
	file, err := os.OpenFile(infile, os.O_RDONLY, os.ModePerm)
	if err != nil {
		return err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	BUF := 10000
	collection := make([]*Rating, 0, BUF)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return err
			} else {
				break
			}
		}
		arr := strings.Split(strings.Trim(line, "\n"), " ")
		if len(arr) >= 2 {
			if uid, err := strconv.ParseInt(arr[0], 10, 64); err != nil {
				glog.Errorf("parse uid failed: %s", arr[0])
				continue
			} else {
				for _, ele := range arr[1:] {
					brr := strings.Split(ele, ":")
					if len(brr) == 2 {
						if itemid, err := strconv.ParseInt(brr[0], 10, 64); err != nil {
							glog.Errorf("parse itemid failed: %s", ele)
							continue
						} else {
							if rate, err := strconv.ParseFloat(brr[1], 64); err != nil {
								glog.Errorf("parse rate failed: %s", brr[1])
								continue
							} else {
								collection = append(collection, &Rating{
									Uid:    uid,
									ItemId: itemid,
									Rate:   float32(rate * decayCoef),
								})
								if len(collection) >= BUF {
									for _, inst := range Shuffle(collection) { //打乱样本的顺序，否则可能一直在更新某一个item对应的参数，导致浮点数溢出
										channel <- inst
									}
									collection = make([]*Rating, 0, BUF)
								}
							}
						}
					}
				}
			}
		}
	}
	if len(collection) > 0 {
		for _, inst := range Shuffle(collection) {
			channel <- inst
		}
	}
	return nil
}

func (self *SVD) update(rating *Rating) {
	uid := rating.Uid
	itemid := rating.ItemId
	rate := rating.Rate
	rate_hat, userIndex, itemIndex := self.Predict(uid, itemid)
	if userIndex >= 0 && itemIndex >= 0 {
		err := rate - rate_hat
		self.addError(err)
		for f := 0; f < self.F; f++ {
			deltaP := err*self.Q[itemIndex][f] - self.lambda*self.P[userIndex][f]
			self.P[userIndex][f] += self.learningRate * deltaP
		}
		//把P向量更新完再去更新Q向量
		for f := 0; f < self.F; f++ {
			deltaQ := err*self.P[userIndex][f] - self.lambda*self.Q[itemIndex][f]
			self.Q[itemIndex][f] += self.learningRate * deltaQ
		}
		deltaBu := err - self.lambda*self.Bu[userIndex]
		self.Bu[userIndex] += self.learningRate * deltaBu
		deltaBi := err - self.lambda*self.Bi[itemIndex]
		self.Bi[itemIndex] += self.learningRate * deltaBi
	}
}

func (self *SVD) SaveModel(modelFile string) {
	if fo, err := os.OpenFile(modelFile, os.O_TRUNC|os.O_CREATE|os.O_RDWR, os.ModePerm); err != nil {
		glog.Errorf("create file %f failed %v", modelFile, err)
		return
	} else {
		defer fo.Close()
		enc := gob.NewEncoder(fo)
		if err := enc.Encode(*self); err != nil {
			glog.Errorf("encode model failed %v", err)
			return
		}
	}
}

func LoadSVDModel(modelFile string) (*SVD, error) {
	if fi, err := os.OpenFile(modelFile, os.O_RDONLY, os.ModePerm); err != nil {
		glog.Errorf("create file %f failed %v", modelFile, err)
		return nil, err
	} else {
		defer fi.Close()
		var model SVD
		dec := gob.NewDecoder(fi)
		if err := dec.Decode(&model); err != nil {
			glog.Errorf("decode model failed %v", err)
			return nil, err
		} else {
			glog.Infof("load model from file %s", modelFile)
			return &model, nil
		}
	}
}

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

