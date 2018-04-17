package main

import (
	"github.com/Nebula1084/mercury/server/pricer"
	"fmt"
)

func printResult(result *pricer.Result) {
	fmt.Printf("mean: %f, conf :%f\n", result.Mean, result.Conf)
}

func testEuropean() {
	fmt.Println("European:")
	european := pricer.European{
		Interest: 0.06,
		Repo:     0,
		Asset: pricer.Asset{
			Price:      100,
			Volatility: 0.1,
		},
		Instrument: pricer.Instrument{
			Maturity:   3,
			Strike:     100,
			OptionType: pricer.CALL,
		},
	}
	printResult(european.Calculate())
}

func testVolatility() {
	fmt.Println("Volatility:")
	volatility := pricer.Volatility{
		Interest: 0.04,
		Repo:     0.2,
		Instrument: pricer.Instrument{
			Maturity:   0.0219,
			Strike:     1.8,
			OptionType: pricer.CALL,
		},
		Premium: 0.1547,
		Price:   1.9595,
	}
	printResult(volatility.Calculate())
}

func testAmerican() {
	fmt.Println("American:")
	american := pricer.American{
		Interest: 0.06,
		Asset: pricer.Asset{
			Price:      100,
			Volatility: 0.1,
		},
		Instrument: pricer.Instrument{
			Maturity:   3,
			Strike:     100,
			OptionType: pricer.CALL,
		},
		Step:   5000,
		UseGpu: false,
	}
	printResult(american.Calculate())
	american.UseGpu = true
	american.Instrument.OptionType = pricer.PUT
	printResult(american.Calculate())
}

func testGeometricEuropean() {
	fmt.Println("Geometric European:")
	geometricEuropean := pricer.GeometricEuropean{
		ClosedForm: true,
		UseGpu:     false,
		BasketSize: 2,
		Interest:   0.05,
		PathNum:    1e6,
		Instrument: pricer.Instrument{
			Maturity:   3,
			Strike:     100,
			OptionType: pricer.CALL,
		},
		Assets: []pricer.Asset{
			{Price: 100, Volatility: 0.3, Mean: -1},
			{Price: 100, Volatility: 0.3, Mean: -1},
		},
		CorMatrix: []float64{
			1, 0.5,
			0.5, 1,
		},
	}

	printResult(geometricEuropean.Calculate())
	geometricEuropean.ClosedForm = false
	geometricEuropean.UseGpu = true
	printResult(geometricEuropean.Calculate())
	geometricEuropean.UseGpu = false
	printResult(geometricEuropean.Calculate())
}

func testArithmeticEuropean() {
	fmt.Println("Arithmetic European:")
	arithmetic := pricer.ArithmeticEuropean{
		ControlVariate: false,
		UseGpu:         false,
		BasketSize:     2,
		Interest:       0.05,
		PathNum:        1e6,
		Instrument: pricer.Instrument{
			Maturity:   3,
			Strike:     100,
			OptionType: pricer.CALL,
		},
		Assets: []pricer.Asset{
			{Price: 100, Volatility: 0.3, Mean: -1},
			{Price: 100, Volatility: 0.3, Mean: -1},
		},
		CorMatrix: []float64{
			1, 0.5,
			0.5, 1,
		},
	}

	printResult(arithmetic.Calculate())
	arithmetic.ControlVariate = true
	printResult(arithmetic.Calculate())
	arithmetic.UseGpu = true
	printResult(arithmetic.Calculate())
}

func testGeometricAsian() {
	fmt.Println("Geometric Asian:")
	asian := pricer.GeometricAsian{
		ClosedForm:  true,
		UseGpu:      false,
		Interest:    0.05,
		PathNum:     1e5,
		Observation: 50,
		Asset: pricer.Asset{
			Price: 100, Volatility: 0.3, Mean: -1,
		},
		Instrument: pricer.Instrument{
			Maturity: 3, Strike: 100, OptionType: pricer.CALL,
		},
	}
	printResult(asian.Calculate())
	asian.ClosedForm = false
	printResult(asian.Calculate())
	asian.UseGpu = true
	printResult(asian.Calculate())
}

func testArithmeticAsian() {
	fmt.Println("Arithmetic Asian:")
	asian := pricer.ArithmeticAsian{
		ControlVariate: false,
		UseGpu:         false,
		Interest:       0.05,
		PathNum:        1e5,
		Observation:    50,
		Asset: pricer.Asset{
			Price: 100, Volatility: 0.3, Mean: -1,
		},
		Instrument: pricer.Instrument{
			Maturity: 3, Strike: 100, OptionType: pricer.CALL,
		},
	}
	printResult(asian.Calculate())
	asian.ControlVariate = true
	printResult(asian.Calculate())
	asian.Instrument.OptionType = pricer.PUT
	asian.UseGpu = true
	printResult(asian.Calculate())
}

func main() {
	testEuropean()
	testVolatility()
	testAmerican()
	testGeometricEuropean()
	testArithmeticEuropean()
	testGeometricAsian()
	testArithmeticAsian()
}
