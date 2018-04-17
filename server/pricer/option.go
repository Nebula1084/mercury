package pricer

type Option interface {
	Calculate() *Result
}

func Call(protocol *Protocol) *Result {
	res, err := protocol.Call()
	if err != nil {
		return &Result{Mean: -1, Conf: -1}
	} else {
		return res
	}
}

type European struct {
	Interest   float64
	Repo       float64
	Instrument Instrument
	Asset      Asset
}

func (e *European) Calculate() *Result {
	var basketSize int8
	basketSize = 1
	assets := make([]Asset, basketSize)
	corMatrix := make([]float64, basketSize*basketSize)
	assets[0] = e.Asset
	corMatrix[0] = 1

	protocol := Protocol{
		Op:         EUROPEAN,
		Interest:   e.Interest,
		Repo:       e.Repo,
		Instrument: e.Instrument,
		BasketSize: basketSize,
		Assets:     assets,
		CorMatrix:  corMatrix,
	}
	return Call(&protocol)
}

type Volatility struct {
	Interest   float64
	Repo       float64
	Instrument Instrument
	Premium    float64
	Price      float64
}

func (v *Volatility) Calculate() *Result {
	assets := make([]Asset, 1)
	assets[0] = Asset{Price: v.Price}

	protocol := Protocol{
		Op:         VOLATILITY,
		Interest:   v.Interest,
		Repo:       v.Repo,
		Instrument: v.Instrument,
		Premium:    v.Premium,
		Assets:     assets,
	}

	return Call(&protocol)
}

type GeometricEuropean struct {
}

type ArithmeticEuropean struct {
}

type American struct {
	Interest   float64
	Instrument Instrument
	Asset      Asset
	Step       int32
}

func (a *American) Calculate() *Result {
	return nil
}

type GeometricAsian struct {
}
type ArithmeticAsian struct {
}
