#include <iostream>
using namespace std;
int main()
{
    long long K ;
    cin >> K ;

    long long a = 0;
    long long b = 1;
    if( K >= 0 ){
        while( b <= K){
            long long temp = b;
            b = a + b ;
            a = temp;
        }
    }else{
        b = 0;
    }

    cout << b;
    return 0;
}

