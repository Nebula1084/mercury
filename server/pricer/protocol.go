package pricer

import (
	"bytes"
	"encoding/binary"
	"math"
)

type OptionType int8

type Operation int8

type Asset struct {
	Price      float64
	Mean       float64
	Volatility float64
}

type Instrument struct {
	Maturity   float64
	Strike     float64
	OptionType OptionType
}

const (
	EUROPEAN            Operation = 1
	VOLATILITY          Operation = 2
	AMERICAN            Operation = 3
	GEOMETRIC_EUROPEAN  Operation = 4
	ARITHMETIC_EUROPEAN Operation = 5
	GEOMETRIC_ASIAN     Operation = 6
	ARITHMETIC_ASIAN    Operation = 7

	CALL OptionType = 1
	PUT  OptionType = 2
)

type Protocol struct {
	Op             Operation
	Interest       float64
	Repo           float64
	Premium        float64
	Instrument     Instrument
	Step           int32
	PathNum        int32
	ClosedForm     int8
	UseGpu         int8
	ControlVariate int8
	BasketSize     int8
	Assets         []Asset
	CorMatrix      []float64
}

type Result struct {
	Mean float64
	Conf float64
}

func MakeResult(buff []byte) *Result {
	result := new(Result)
	retBits := binary.LittleEndian.Uint64(buff[:8])
	result.Mean = math.Float64frombits(retBits)
	retBits = binary.LittleEndian.Uint64(buff[8:])
	result.Conf = math.Float64frombits(retBits)

	return result
}

func (p *Protocol) Bytes() []byte {
	var buff bytes.Buffer
	binary.Write(&buff, binary.LittleEndian, p.Op)
	binary.Write(&buff, binary.LittleEndian, p.Interest)
	binary.Write(&buff, binary.LittleEndian, p.Repo)
	binary.Write(&buff, binary.LittleEndian, p.Premium)
	binary.Write(&buff, binary.LittleEndian, p.Instrument)
	binary.Write(&buff, binary.LittleEndian, p.Step)
	binary.Write(&buff, binary.LittleEndian, p.PathNum)
	binary.Write(&buff, binary.LittleEndian, p.ClosedForm)
	binary.Write(&buff, binary.LittleEndian, p.UseGpu)
	binary.Write(&buff, binary.LittleEndian, p.ControlVariate)
	binary.Write(&buff, binary.LittleEndian, p.BasketSize)

	var i int8
	for i = 0; i < p.BasketSize; i++ {
		binary.Write(&buff, binary.LittleEndian, p.Assets[i])
	}

	for i = 0; i < p.BasketSize*p.BasketSize; i++ {
		binary.Write(&buff, binary.LittleEndian, p.CorMatrix[i])
	}

	return buff.Bytes()
}

func (p *Protocol) Call() (*Result, error) {

	ipc, err := NewIPC()

	if err != nil {
		print(err.Error())
		return nil, err
	}
	ipc.Write(p.Bytes())

	retBytes := make([]byte, 16)
	ipc.Read(retBytes)

	return MakeResult(retBytes), err

}
