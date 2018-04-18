import React, { PropTypes } from 'react'
import Markdown from 'react-remarkable'
import { connect } from 'dva'

export default class Introduction extends React.Component {

    render() {
        return (
            <div>

                <Markdown>{`
# Introduction
## Architecture
2. Implement Front-end with React, which will be packed as static files.
3. Server is implemented by Go and host the static files and HTTP API with Echo
4. In order to take advantage of GPU, we implement pricer with C++ and CUDA.
5. Go sever communicate with pricer by Unix Socket IPC.

## Monte-Carlo with GPU
1. Cholesky Decomposition
2. Multi Thread Sampling
Allocate sampling task and memory for every thread, so that they can sampling in parallel.
O(N/P)

3. Sum Reduction
Every two elements add together. Only O(Log(N)) time complexity with sufficient processor.
O(N/P+Log(P)) P is GPU core number
## Binomial Tree with GPU
1. Multi Thread Iteration
Improve to O(N) ?

NO, O(N^2/P)
2. Double buffer swap
A[i]=F(A[i],A[i+1])
Such synchronization method only exists within one block.  
To avoid synchronization
B[i]=F(A[i],A[i+1])
swap the pointer after each iteration.

        `}</Markdown>
            </div>
        )
    }
}