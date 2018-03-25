package main

import "github.com/Nebula1084/mercury/server/pricer"

func main() {
	option := pricer.European{
		Interest: 0.05,
		Repo:     0.1,
		Instrument: pricer.Instrument{
			Maturity:   0.2,
			Strike:     123,
			OptionType: pricer.CALL,
		},
		Asset: pricer.Asset{
			Price:      12.2,
			Volatility: 0.2,
		},
	}
	res, _ := option.Calculate()
	println(res)
	vol := pricer.Volatility{
		Interest: 0.06,
		Repo:     0.2,
		Instrument: pricer.Instrument{
			Maturity:   0.4,
			Strike:     233,
			OptionType: pricer.PUT,
		},
		Price: 234,
	}
	res, _ = vol.Calculate()
	println(res)
}
