package pricer

func main() {
	option := European{
		Interest: 0.05,
		Repo:     0.1,
		Instrument: Instrument{
			Maturity:   0.2,
			Strike:     123,
			OptionType: CALL,
		},
		Asset: Asset{
			Price:      12.2,
			Volatility: 0.2,
		},
	}
	res, _ := option.Calculate()
	println(res)
	vol := Volatility{
		Interest: 0.06,
		Repo:     0.2,
		Instrument: Instrument{
			Maturity:   0.4,
			Strike:     233,
			OptionType: PUT,
		},
		Price: 234,
	}
	res, _ = vol.Calculate()
	println(res)
}
