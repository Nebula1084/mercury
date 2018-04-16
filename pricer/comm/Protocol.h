#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <option/Option.h>

typedef char Operation;

const Operation EUROPEAN = 1;
const Operation VOLATILITY = 2;
const Operation AMERICAN = 3;
const Operation GEOMETRIC_EUROPEAN = 4;
const Operation ARITHMETIC_EUROPEAN = 5;
const Operation GEOMETRIC_ASIAN = 6;
const Operation ARITHMETIC_ASIAN = 7;

struct Protocol
{
    Operation operation;
    double interest;
    double repo;
    double premium;
    Instrument instrument;
    int step; // observation in Asian
    int pathNum;
    char closedForm;
    char useGpu;
    char controlVariate;
    char basketSize;
    Asset asset;

    static Option *parse(Protocol *buff);
} __attribute__((packed));
#endif