#include <iostream>
using namespace std;

const int Max = 1000000007;
long long a[100000];

void Day_so(long long n){
	for(int i = 1; i<=n; i++)
	{
		if (i==1) 
			a[i] = 0;
		else if (i==2 || i ==3) 
			a[i] = 1;
		else if (i==4) 
			a[i] = 2;
		else
			a[i] = (a[i-2]%Max) + (a[i-3]%Max) + (a[i-4]% Max);
	}
	cout << a[n] % Max;
}

int main()
{
	long long n;
	cin >> n;
	
	Day_so(n);

return 0;
}

