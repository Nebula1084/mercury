#include <comm/IPC.h>
#include <comm/Protocol.h>

void *price(void *message)
{
    int *connFd = (int *)message;
    char recvBuf[2048];

    int size = recv(*connFd, recvBuf, 1000, 0);

    if (size >= 0)
    {
        Option *task = Protocol::parse((Protocol *)recvBuf);
        Result result = task->calculate();
        delete task;
        send(*connFd, &result, sizeof(Result), 0);
    }
    if (size == -1)
    {
        printf("Error[%d] when receiving Data:%s.\n", errno, strerror(errno));
    }

    free(message);
    close(*connFd);
}

int main()
{
    IPC server("pricer.ipc");
    server.start(price);
    return 0;
}