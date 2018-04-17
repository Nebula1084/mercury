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

func boolToByte(b bool) int8 {
	if b {
		return 1
	} else {
		return 0
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
	var basketSize int8
	basketSize = 1
	assets := make([]Asset, basketSize)
	corMatrix := make([]float64, basketSize*basketSize)
	assets[0] = Asset{Price: v.Price}
	corMatrix[0] = 1

	protocol := Protocol{
		Op:         VOLATILITY,
		Interest:   v.Interest,
		Repo:       v.Repo,
		Instrument: v.Instrument,
		Premium:    v.Premium,
		BasketSize: basketSize,
		Assets:     assets,
		CorMatrix:  corMatrix,
	}

	return Call(&protocol)
}

type American struct {
	Interest   float64
	Instrument Instrument
	Asset      Asset
	Step       int32
	UseGpu     bool
}

func (a *American) Calculate() *Result {
	var basketSize int8
	basketSize = 1
	assets := make([]Asset, basketSize)
	corMatrix := make([]float64, basketSize*basketSize)
	assets[0] = a.Asset
	corMatrix[0] = 1
	useGpu := boolToByte(a.UseGpu)

	protocol := Protocol{
		Op:         AMERICAN,
		Interest:   a.Interest,
		Instrument: a.Instrument,
		Step:       a.Step,
		UseGpu:     useGpu,
		BasketSize: basketSize,
		Assets:     assets,
		CorMatrix:  corMatrix,
	}
	return Call(&protocol)
}

type GeometricEuropean struct {
	ClosedForm bool
	UseGpu     bool
	BasketSize int8
	Interest   float64
	Instrument Instrument
	Assets     []Asset
	CorMatrix  []float64
	PathNum    int32
}

func (ge *GeometricEuropean) Calculate() *Result {
	closedForm := boolToByte(ge.ClosedForm)
	useGpu := boolToByte(ge.UseGpu)
	protocol := Protocol{
		Op:         GEOMETRIC_EUROPEAN,
		ClosedForm: closedForm,
		UseGpu:     useGpu,
		BasketSize: ge.BasketSize,
		Interest:   ge.Interest,
		Instrument: ge.Instrument,
		Assets:     ge.Assets,
		CorMatrix:  ge.CorMatrix,
		PathNum:    ge.PathNum,
	}
	return Call(&protocol)
}

type ArithmeticEuropean struct {
	ControlVariate bool
	UseGpu         bool
	BasketSize     int8
	Interest       float64
	Instrument     Instrument
	Assets         []Asset
	CorMatrix      []float64
	PathNum        int32
}

func (ae *ArithmeticEuropean) Calculate() *Result {
	controlVariate := boolToByte(ae.ControlVariate)
	useGpu := boolToByte(ae.UseGpu)
	protocol := Protocol{
		Op:             ARITHMETIC_EUROPEAN,
		ControlVariate: controlVariate,
		UseGpu:         useGpu,
		BasketSize:     ae.BasketSize,
		Interest:       ae.Interest,
		Instrument:     ae.Instrument,
		Assets:         ae.Assets,
		CorMatrix:      ae.CorMatrix,
		PathNum:        ae.PathNum,
	}
	return Call(&protocol)
}

type GeometricAsian struct {
	ClosedForm  bool
	UseGpu      bool
	Asset       Asset
	Interest    float64
	Instrument  Instrument
	PathNum     int32
	Observation int32
}

func (ga *GeometricAsian) Calculate() *Result {
	closedForm := boolToByte(ga.ClosedForm)
	useGpu := boolToByte(ga.UseGpu)
	protocol := Protocol{
		Op:         GEOMETRIC_ASIAN,
		ClosedForm: closedForm,
		UseGpu:     useGpu,
		Assets:     []Asset{ga.Asset},
		Interest:   ga.Interest,
		Instrument: ga.Instrument,
		PathNum:    ga.PathNum,
		Step:       ga.Observation,
	}
	return Call(&protocol)

}

type ArithmeticAsian struct {
	ControlVariate bool
	UseGpu         bool
	Asset          Asset
	Interest       float64
	Instrument     Instrument
	PathNum        int32
	Observation    int32
}

func (aa *ArithmeticAsian) Calculate() *Result {
	controlVariate := boolToByte(aa.ControlVariate)
	useGpu := boolToByte(aa.UseGpu)
	protocol := Protocol{
		Op:             ARITHMETIC_ASIAN,
		ControlVariate: controlVariate,
		UseGpu:         useGpu,
		Assets:         []Asset{aa.Asset},
		Interest:       aa.Interest,
		Instrument:     aa.Instrument,
		PathNum:        aa.PathNum,
		Step:           aa.Observation,
	}
	return Call(&protocol)
}
