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
    return { ...state }
}

export function cancel(state, { payload: index }) {
    state.rows[index] = { ...state.stash[index], editing: false };
    delete state.stash[index];
    return { ...state }
}