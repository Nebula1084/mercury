#include <option/European.h>
#include <option/Volatility.h>
#include <IPC.h>

void *price(void *message)
{
    int *conn_fd = (int *)message;
    char recv_buf[2048];
    European european;

    int size = recv(*conn_fd, recv_buf, 1000, 0);
    float res;
    European *option;
    if (size >= 0)
    {
        Operation *op = (Operation *)recv_buf;
        switch (*op)
        {
        case EUROPEAN:
            res = ((European *)(recv_buf + 1))->calculate();
            break;
        case VOLATILITY:
            res = ((Volatility *)(recv_buf + 1))->calculate();
            break;
        case AMERICAN:
            res = ((European *)(recv_buf + 1))->calculate();
            break;
        }

        send(*conn_fd, &res, 4, 0);
    }
    if (size == -1)
    {
        printf("Error[%d] when receiving Data:%s.\n", errno, strerror(errno));
    }

    free(message);
    close(*conn_fd);
}

int main()
{
    IPC server("pricer.ipc");
    server.start(price);
    return 0;
}