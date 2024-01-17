#include <bits/stdc++.h>
using namespace std;
int countVector(vector<int>a){
	sort(a.begin(),a.end());
	int dem = 0;
	vector<vector<int> > b;
	vector<int>c;
	for(int i=0;i<a.size();i++){
		for(int j=0;j<b.size();j++){
			if(b[j].size()<5 && (b[j][b[j].size()-1]-b[j][b[j].size()-2]==a[i]-b[j][b[j].size()-1])){
				b[j].push_back(a[i]);
			}
		}
		for(int j=0;j<c.size();j++){
			vector<int>d;
			d.push_back(c[j]);
			d.push_back(a[i]);
			b.push_back(d);
		}
		if(i<=(a.size()-5))
			if((a[i+1]-a[i])*5<=a[a.size()-1])
				c.push_back(a[i]);
		if(i<a.size()-1){
			if(a[i]==a[i+1])
				dem++;
		}
	}
	for(int i =0;i<b.size();i++){
		if(b[i].size()==5)
			dem++;
	}
	return dem;
}
int main(){
	vector<int> a;
	int n;
	cin >>n;
	for(int i=0;i<n;i++){
		int x;
		cin >>x;
		a.push_back(x);
	}
	cout <<countVector(a);
}
