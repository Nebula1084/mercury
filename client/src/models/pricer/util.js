function toCamelCase(str) {
    let ret = str.split(' ')
    ret = ret.join('');
    ret = ret[0].toLowerCase() + ret.slice(1);
    return ret;
}

export function insert(state, rows, editing) {
    let base = state.rows.length;

    for (let i = 0; i < rows.length; i++) {
        rows[i].key = base + i;
        rows[i].editing = editing
    }
    return { ...state, rows: [...state.rows, ...rows] };
}

export function load(state, { payload }) {
    return insert(state, payload, false)
}

export function add(state, { payload }) {
    return insert(state, [payload], false)
}

export function update(state, { payload: { value, index, column } }) {
    let record = state.rows[index]
    record[column] = value;
    return { ...state };
}

export function edit(state, { payload: index }) {
    let record = state.rows[index];
    state.stash[index] = { ...record }
    record.editing = true;
    return { ...state }
}

export function save(state, { payload: index }) {
    let record = state.rows[index];
    record.editing = false;
    delete state.stash[index]
    delete state.rows[index].price
    return { ...state }
}

export function cancel(state, { payload: index }) {
    state.rows[index] = { ...state.stash[index], editing: false };
    delete state.stash[index];
    return { ...state }
}

export function alter(state) {
    let stock = "Stock Price " + state.stockNum;
    let vol = "Volatility " + state.stockNum
    let num = state.stockNum;
    let idx = 0;

    let dataIndices = []
    dataIndices.push(toCamelCase(stock))
    dataIndices.push(toCamelCase(vol))

    let newColumns = [];
    for (let i = 0; i < num * 2; i++) {
        newColumns.push(state.columns[idx]);
        idx += 1;
    }
    newColumns.push(stock);
    newColumns.push(vol)
    for (let i = 0; i < (num - 1) * num / 2; i++) {
        newColumns.push(state.columns[idx])
        idx += 1;
    }
    for (let i = 0; i < num; i++) {
        let cor = "Cor " + i + num;
        dataIndices.push(toCamelCase(cor));
        newColumns.push(cor);
    }

    for (let i = idx; i < state.columns.length; i++) {
        newColumns.push(state.columns[i]);
    }

    for (let i = 0; i < state.rows.length; i++) {
        for (let j = 0; j < dataIndices.length; j++) {
            state.rows[i][dataIndices[j]] = '0';
        }
    }
    state.stockNum += 1;
    state.columns = newColumns;
    return { ...state }
}