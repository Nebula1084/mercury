package main

import (
	"github.com/Nebula1084/mercury/server/pricer"
	"fmt"
)

func printResult(result *pricer.Result) {
	fmt.Printf("mean: %f, conf :%f\n", result.Mean, result.Conf)
}

func testEuropean() {
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

func main() {
	testEuropean()
}
