export function add(state, { payload: rows }) {
    let base = state.rows.length;

    for (let i = 0; i < rows.length; i++) {
        rows[i].key = base + i;
    }
    return { rows: [...state.rows, ...rows] };
}