#include <iostream>
using namespace std;
int main()
{
    long long k ;
    //cout << "Nhap so k = ";
    cin >> k ;

    long long f0 = 0;
    long long f1 = 1;
    if( k >= 0 ){
        while( f1 <= k){
            long long temp = f1;
            f1 += f0 ;
            f0 = temp;
        }
    }else{
        f1 = 0;
    }
	//cout << "So fibonacci nho nhat lon hon k la: " << f1;
    cout << f1;
    return 0;
}

