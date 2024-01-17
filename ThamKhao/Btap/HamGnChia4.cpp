#include <iostream>
#include<map>
using namespace std;

/*
g(1) = 1
g(3) = 3
g(2n) = g(n)
g(4n + 1) = 2g(2n+1) - gn
g(4n + 3) = 3g(2n+1) - 2g(n)

	
n = 1 => g(2) = g1 = 1
		 g(5) = 2 * g3 - g1 = 5
 		g(7) = 3*g3 - 2*g1 = 7
 		
n = 2 => g(4) = g2 = 1
		g(9) = 2 * g5 - g2 = 9
		g(11) = 3*g5 - 2*g2 = 13

n = 3 => g(6) = g3 = 3
		g(13) = 2 * g7 - g3 = 11
		g(15) = 3*g7 - 2*g3 = 15

n = 4 => g(8) = g4 = 1
n = 5 => g(10) = g5 = 5
*/

long long fiboG (long long n)
{
	long long temp;
	
	if(n == 1)
		return 1;
	if (n == 3)
		return 3;
		
	if (n % 2 == 0)
	{
		temp = n/2;
		return fiboG(temp);
	}
	
	if (n % 4 == 1) {
		temp = (n-1)/4;
		return 2*fiboG(2*temp+1) - fiboG(temp);
	}
	
	if (n % 4 == 3) {
		temp = (n-3)/4;
		return 3*fiboG(2*temp+1) - 2*fiboG(temp);
	}
		
}
int main()
{
	long long n;
	cout << "Nhap n = "; cin >> n;
	while(n <= 0) {
		cout << "Nhap n = "; cin >> n;
	}
	cout << "g(" << n << ") = " << fiboG(n);
return 0;
}

