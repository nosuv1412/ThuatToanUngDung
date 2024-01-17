#include <iostream>
#include <cmath>
using namespace std;

int print(int n) {
	if(n==1) cout << 1;
	else{
		print(n-1);
		cout << " " << n ;
	}
}
int main()
{
	 print(50);

return 0;
}

