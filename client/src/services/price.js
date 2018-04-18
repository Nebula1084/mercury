import { config, get, post } from 'utils'

export const EUROPEAN = 1
export const VOLATILITY = 2
export const AMERICAN = 3
export const GEOMETRIC_EUROPEAN = 4
export const ARITHMETIC_EUROPEAN = 5
export const GEOMETRIC_ASIAN = 6
export const ARITHMETIC_ASIAN = 7

export function buildProtocol(Operation, data) {
    let protocol = {
        Op: Operation
    }
    protocol.Interest = Number(data.interest)
    if (data.repo)
        protocol.Repo = Number(data.repo)

    if (data.premium)
        protocol.premium = Number(data.premium)

    protocol.Instrument = {
        Maturity: Number(data.maturity),
        Strike: Number(data.strike),
        OptionType: Number(data.optionType)
    }

    if (data.step)
        protocol.Step = Number(data.step);
    if (data.pathNum)
        protocol.PathNum = Number(data.pathNum);
    if (data.closedForm != null)
        protocol.ClosedForm = data.closedForm ? 1 : 0;
    if (data.useGpu != null)
        protocol.UseGpu = data.useGpu ? 1 : 0;
    if (data.controlVariate != null)
        protocol.ControlVariate = data.controlVariate ? 1 : 0;

    if (data.basketSize != null)
        protocol.BasketSize = Number(data.basketSize);
    else
        protocol.BasketSize = 1;
    protocol.Assets = []

    const size = protocol.BasketSize;
    for (let i = 0; i < size; i++) {
        let v = -1;
        if (data['volatility' + i] != null)
            v = Number(data['volatility' + i]);
        protocol.Assets.push(
            {
                Price: Number(data['stockPrice' + i]),
                Volatility: v
            }
        );
    }
    if (size == 1)
        protocol.CorMatrix = [1];
    else {
        protocol.CorMatrix = [];
        for (let i = 0; i < size; i++)
            for (let j = 0; j < size; j++) {
                if (i == j) {
                    protocol.CorMatrix.push(1);
                } else {
                    let idx = 'cor'
                    if (i < j) {
                        idx += i;
                        idx += j;
                    } else {
                        idx += j;
                        idx += i;
                    }

                    protocol.CorMatrix.push(Number(data[idx]))
                }
            }
    }
    return protocol
}

export async function pricing(params) {
    let result = post(`${config.api}price`, params)
    return result
}

export function geometric() {

}