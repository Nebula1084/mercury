package pricer

import (
	"encoding/binary"
	"bytes"
	"math"
)

type Asset struct {
	Price      float32
	Volatility float32
}

type Instrument struct {
	Maturity   float32
	Strike     float32
	OptionType OptionType
}

type OptionType int8

type Operation int8

const (
	CALL OptionType = 1
	PUT  OptionType = 2

	EUROPEAN   Operation = 1
	VOLATILITY Operation = 2
	AMERICAN   Operation = 3
)

func makeCall(op Operation, i interface{}) (float32, *IPC, error) {
	ipc, err := NewIPC()
	if err != nil {
		return 0, nil, err
	}
	var buff bytes.Buffer
	binary.Write(&buff, binary.LittleEndian, op)
	binary.Write(&buff, binary.LittleEndian, i)
	err = ipc.Write(buff.Bytes())
	if err != nil {
		ipc.Close()
		return 0, ipc, err
	}
	retBytes := make([]byte, 4)
	ipc.Read(retBytes)
	retBits := binary.LittleEndian.Uint32(retBytes)
	result := math.Float32frombits(retBits)
	return result, ipc, nil
}

type European struct {
	Interest   float32
	Repo       float32
	Instrument Instrument
	Asset      Asset
}

func (e *European) Calculate() (res float32, err error) {
	res, ipc, err := makeCall(EUROPEAN, e)
	ipc.Close()
	return
}

type Volatility struct {
	Interest   float32
	Repo       float32
	Instrument Instrument
	Price      float32
}

func (v *Volatility) Calculate() (res float32, err error) {
	res, ipc, err := makeCall(VOLATILITY, v)
	ipc.Close()
	return
}

type American struct {
	Interest   float32
	Instrument Instrument
	Asset      Asset
	Step       int32
}

func (a *American) Calculate() (res float32, err error) {
	res, ipc, err := makeCall(AMERICAN, a)
	ipc.Close()
	return
}
