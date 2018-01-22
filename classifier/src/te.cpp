#include <iostream>
#include <fstream>
#include <vector>
using namespace :: std;
int main () {

vector<int> d1,d2;
int e,f,i=1,lines;

ifstream read("test.txt");

while(read>>e){
d1.push_back(e);
}
lines=d1.size();

cout<<d1[0]+1<<endl;

return 0;
}