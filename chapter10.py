# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:36:11 2020
@author: Abindu's PC
CTCI: Chapter 10 Questions : Sorting and Searching
Started again: March 18 '20
TODO: Medium and Hard Problems
"""
import sys

# Q1: Sorted Merge
def sortedMerge(a, b):
    """If you start populating from smallest elem you will need temp
    array/ds. Start from end ie fill extra spaces in array a first
    """
    m, n = len(a)-len(b)-1, len(b)-1
    curr = len(a)-1
    while m>=0 and n>=0:
        if a[m]>=b[n]:
            a[curr] = a[m]
            m-=1
        else:
            a[curr] = b[n]
            n-=1
        curr-=1
    # At this point if m>=0 then nothing to be done!!
    while n>=0:
        a[curr] = a[n]
        n-=1
        curr-=1
    return a
            
"""
a = [1,4,7,9,0,0]
b = [-1,12]
print(sortedMerge(a,b))

"""

# Q2: Group Anagrams
from functools import cmp_to_key
from collections import defaultdict

def groupAnagrams1(arr):
    """Size of arr: N and size of largest str is K
    Time: O(N * logN * (K logK) )
    """
    def mycmp(a,b):
        """
        Time: O(K logK + K) ==> ~O(K logK)
        """
        a = ''.join(sorted(a)) # O(K logK)
        b = ''.join(sorted(b)) # O(K logK)
        if a<b: # O(K)
            return 1
        elif a>b:
            return -1
        else:
            return 0
    arr.sort(key=cmp_to_key(mycmp)) # O(N logN * Complexity of compare fn)
    return arr
    
def groupAnagrams2(arr):
    """
    We don't need any overall order. Avoid sorting the entire list of strs
    Sort individual string, use this as the key, this way any anagram group can be collected
    Time: O(N * KlogK)
    Space: O(N*K) N strings of max len K each
    """
    ans = []
    mymap = defaultdict(list)
    for s in arr:
        mymap[''.join(sorted(s))].append(s)
    #print(mymap)
    for k in mymap:
        ans.extend(mymap[k])
    return ans
    
def groupAnagrams3(arr):
    """No sorting at all
    Time: O(N*K)
    Space: O(N * K)
    """
    mymap = defaultdict(list)
    for s in arr:
        chararr = [0]*26
        for c in s:
            chararr[ord(c)-ord('a')]+=1
        mymap[tuple(chararr)].append(s)
    return mymap.values()

"""
arr = ["acre", "aah",  "race", "aha", "care", "haa"]
print(groupAnagrams1(arr.copy()))
print(groupAnagrams2(arr.copy()))
print(groupAnagrams3(arr.copy()))
"""

# Q3: Search in a rotated sorted array
# Skills: Modified Bin Search, Search both sides, Recursion and 
# bubble up!

def searchInRotatedSortedArray(nums, target):
    """Assume array was initially sorted in increasing order
    [5,7,9,12,1,3] 
    
    Tricky Case: [2,2,2,2,2,5,6,2] => When you'd have to search both sides 
    for at least some iterations, below code fails
    """
    n = len(nums)-1
    l, r = 0, n
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]==target:
            return mid
        if nums[l]<nums[mid]:
            #Left is sorted
            if nums[l]<=target<nums[mid]:
                # Search only left half
                r = mid-1
            else:
                # Search only right half
                l = mid+1
        else:
            #Right half is sorted
            if nums[mid]<target<=nums[r]:
                #Search only right half
                l = mid+1
            else:
                # Search only the left half
                r = mid-1
    return -1

# Best implementation!
def searchInRotatedSortedArray2(nums, target):
    """
    This is a more robust implementation, borrowed from CTCI
    Handles the tricky case of repeated elements, and cases where 
    both side search is reqd eg Handles Tricky Case: [2,2,2,2,2,5,6,2]
    """
    return searchHelper(nums, target, 0, len(nums)-1)

#import pdb
def searchHelper(nums, target, lo, hi):
    #pdb.set_trace()
    mid = lo + (hi-lo)//2
    if nums[mid]==target:
        return mid
    if lo<hi:
        if nums[lo]<nums[mid]:
            # Left half sorted
            if nums[lo]<=target<nums[mid]:
                return searchHelper(nums, target, lo, mid-1)
            else:
                return searchHelper(nums, target, mid+1, hi)
        elif nums[mid]<nums[hi]:
            # Right half is sorted
            if nums[mid]<target<=nums[hi]:
                return searchHelper(nums, target, mid+1, hi)
            else:
                return searchHelper(nums, target, lo, mid-1)
        else:
            # Weird case like so
            # [2,2,2,2,2,5,6,2]
            # [2,2,2,2,5,6]
            # [5,6,2,2,2,2]
            if nums[mid]!=nums[hi]:
                # Search right side
                return searchHelper(nums, target, mid+1, hi)
            else:
                # Search both sides
                # Search left side
                res = searchHelper(nums, target, lo, mid-1)
                
                #If not found on left, search right side
                if res==-1:
                    return searchHelper(nums, target, mid+1, hi)
                else:
                    # If left side search worked!
                    return res
    return -1

"""
#nums = [5,7,9,12,1,3] # Simple case
nums = [2,2,2,2,2,5,6,2] # Tricky Case 
target = 5 # Answer should be idx=5
print(searchInRotatedSortedArray(nums, target))
print(searchInRotatedSortedArray2(nums, target))
"""

# Q4: Sorted Search, no size 
# Skills: Modified Binary Search, Searching the right segment

def sortedSearch(Listy, target):
    """
    Remarks: Bad implementation:
    - You don't need to find size
    - Don't need to do Bin Search on entire array
    """
    if Listy.elementAt(0)==target: 
        return 0
    # Ad hoc way of finding size
    idx = 1
    while Listy.elementAt(idx)!=-1:
        idx *= 2
    # Now actual size of List is between idx/2 and idx
    lo, hi = 1, idx
    # Modified binary search
    while lo<=hi:
        mid = lo + (hi-lo)//2
        if target==Listy.elementAt(mid):
            return mid
        if target<Listy.elementAt(mid):
            hi = mid-1
        else:
            lo = mid+1
    return -1

def sortedSearch2(Listy, target):
    """
    Efficient Implementation
    - Better while loop, early stop
    - Better search space, Search only between idx/2 and idx
    - Handle Listy[mid] == -1 ==> Search left
    """
    if Listy.elementAt(0)==target: 
        return 0
    # Stop if listy[idx]>target
    idx = 1
    while Listy.elementAt(idx)!=-1 and Listy.elementAt(idx)<target:
        idx *= 2
    # The target lies between idx/2 and idx
    lo, hi = idx/2, idx
    # Modified binary search
    while lo<=hi:
        mid = lo + (hi-lo)//2
        elem = Listy.elementAt(mid)
        if elem==-1:
            hi = mid-1
        elif target==elem:
            return mid
        elif target<elem:
            hi = mid-1
        else:
            lo = mid+1
    return -1


# Q5: Sparse Search
# Input is sorted but interspersed with empty strings
def sparseSearch(strs, target):
    """
    If strs[mid]=="" ==> Search both sides?
    """
    if not strs or not target: return -1
    return sparseSearchHelper(strs, target, 0, len(strs)-1)

def sparseSearchHelper(strs, target, lo, hi):
    if lo>hi: 
        return -1
    mid = lo + (hi-lo)//2
    # If mid is " " then find first non empty elem 
    # Use Two pointers mid-1 and mid+1, move them along together
    if strs[mid].isspace():
        l, r = mid-1, mid+1 # two pointers, left and right pointers
        while l>=0 and r<=hi:
            if strs[l].isspace() and strs[r].isspace():
                l-=1
                r+=1
            elif not strs[r].isspace():
                mid = r
                break
            else:
                mid = l
                break
    
    # Now the elem at mid is non empty
    if strs[mid]==target:
        return mid
    elif strs[mid]>target:
        return sparseSearchHelper(strs, target, lo, mid-1)
    else:
        return sparseSearchHelper(strs, target, mid+1, hi)

"""
target = "ball"
strs = [" "," ","ball"," ","red"," "," "," "," ","yada"," "," ","yada"]
print(sparseSearch(strs, target))
"""

# Q6: Sort Big File
# 20GB file with one string per line
# RAM requirements => Don't bring in 20GB into RAM

def externalSort(inp_file):
    """
    Read chunk #1 into RAM #file_chunck1
    Sort this chunk => write back to ==> sorted_chunk1
    sorted_chunk2 .......
    Now perform a K-way merge, read Mth part of each file
    merge => to sorted_final_file
    """
    pass

# Q7: Missing Int
# File has 4 Billion Integers, Available Mem 10GB
# Follow up Available memory => 10MB

def missingInteger(inp_file):
    """
    REMARKS: Following discussion is difficult in Python
        
    4 Billion integers => 2^2 * 2^30 Integers => 2^32 Integers
    If all integers were distinct => We need 2^32 bits to map
    what ints we have seen
    
    Available Mem => 1GB => 8* 2^30 bits => 2^33 bits
    
    In CTCI a byte array is used
    In Python we don't have a byte array, we can use an integer array
    
    With 1 GB memory we have 2^33 bits, so we can map 2^32 integers if
    we only consume 1 bit per integer
    How?
    By encoding them in Bit Vector => Create an array of integers
    Python int is 32 bit (assume)
    Use 2^32 //32 integers => They consume 2^32 bits
    bit_vector = [False for _ in range(2**32 //32)] 
    """
    
    # Pseudo-ish code, almost as good as real
    
    # Each integer represents 32 numbers one for each bit
    bit_vector = [0 for _ in range(2**32 /32)]
    
    with open(inp_file) as f:
        curr = f.readnextline()
        """
        curr is the next seen integer
        curr = 30:
        curr//32 => 0th elem in bit_vct curr%32 ==> 30th bit needs to be set
        
        curr = 987654321:
        curr//32 = 30864197 so this is the idx of integer in bit_vector
        Which bit of that integer to be set? => curr%32 => 17
        """
        bit_vector[curr//32] |= (1<< (curr%32))
        
    # Now a bit vector is ready as a seen set hehe!!
    for i in range(len(bit_vector)):
        for j in range(32):
            if bit_vector[i] & (1<<j) == 0:
                return i*32 + j
            
def missingInteger2(inp_file):
    """
    TODO
    If we have 10MB memory only?
    """
    pass

# Q8: Find Duplicates
    
def findDuplicates(nums):
    """
    nums has integers from 1 to N, with duplicates.
    With 4KB memory => convert to bits => 4*8* 1000 => 
    32000 bits => 1000 ints @ 32 bits/4 Bytes per integer
    Also N is at most 32000, so we can have a bit vector to map all nums
    from 1 to 32000
    """
    
    # Create a bit_vector to be used as a map
    bit_vector = [0 for _ in range(1000)]
    
    # bit_vector[0] is an int that maps ints in range 0-31
    for num in nums:
        bit_vector[num//32] |= (1<<num%32)
    
    # Now print duplicates
    for i in range(len(bit_vector)):
        for j in range(32):
            if bit_vector[i] & (1<<j) == 1:
                print(i*32+j)


# Q9: Sorted Matrix Search
                
def searchSortedMatrix(matrix, target):
    """
    Each row and each column is sorted in ascending order
    """
    if not matrix: return False
    rows, cols = len(matrix), len(matrix[0])
    r, c = 0, len(matrix[0])-1
    while 0<=r<rows and 0<=c<cols:
        if matrix[r][c]==target:
            return True, r, c
        elif matrix[r][c]>target:
            c-=1
        else:
            r+=1
    return False

"""
matrix = [[1,   4,  7, 11, 15],
          [2,   5,  8, 12, 19],
          [3,   6,  9, 16, 22],
          [10, 13, 14, 17, 24],
          [18, 21, 23, 26, 30]]
target = 5
print(searchSortedMatrix(matrix, target))
"""        
        
# Q10: Rank from Stream
# [BST]: Actually going to implement my first ever BST LULZ k
# TODO: How is this diff from full fledged BST??
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left_count = 0
        self.left = None
        self.right = None
        
    def insert(self, new_val):
        """
        Algo: Compare new_val to root's value and insert at correct place
        Simultaneously update self.left_count
        """
        if new_val<=self.val:
            if not self.left:
                self.left = TreeNode(new_val)
            else:
                self.left.insert(new_val)
            self.left_count+=1
        else:
            if not self.right:
                self.right = TreeNode(new_val)
            else:
                self.right.insert(new_val)
                
    def getrank(self, num):
        """
        Perform an inorder traversal, keep the _sum of left_count
        """
        if num==self.val:
            return self.left_count
        elif num<self.val:
            # num is < root.val
            # Have to go left
            if self.left:
                return self.left.getrank(num)
            else:
                return -1
        else:
            # num is > root.val
            # Have to go right but also accumulate left_count
            if self.right:
                res = self.right.getrank(num)
                if res==-1:
                    return -1
                else:
                    return res+self.left_count+1
                  

class Stream:
    def __init__(self):
        self.root = None
        
    def track(self, num):
        if self.root is None:
            self.root = TreeNode(num)
        else:
            self.root.insert(num)
    
    def getRank(self, num):
        return self.root.getrank(num)

"""Working well
s = Stream()
for num in [5,1,4,4,5,9,7,13,3]:
    s.track(num)
for num in [1,3,4]:
    print(s.getRank(num))
"""

# Q11: Peaks and Valleys

def peaksAndValleys(nums):
    """
    O(nlogn) Simple Algo
    Sort the array, then swap pairs ie 0,1 then 2,3 then 4,5 ....
    - [0,1,4,7,8,9]
    - [1,0,4,7,8,9] swap 0,1
    - [1,0,7,4,8,9] swap 4,7
    .
    .
    .
    """
    nums.sort()
    # [0,1,4,7,8,9]
    # idx=1 then swap nums[0], nums[1]
    # idx+=2
    idx = 1
    while idx<len(nums):
        # Swap
        nums[idx-1], nums[idx] = nums[idx], nums[idx-1]
        idx+=2
    return nums

def peaksAndValleys2(nums):
    """
    Main Idea: No sorting, Take a window of 3 elements
    Swap middle with largest of remaining, will ensure a peak forms
    - [0,1,4,7,8,9]
    - [0,4,1,7,8,9] # [0,1,4,...] => [0,4,1,...]
    - [0,4,1,7,9,8]
    """
    mid = 1
    while mid<len(nums):
        makeMiddlePeak(nums, mid)
        mid+=2
    return nums

def makeMiddlePeak(nums, mid):
    _max = max(nums[mid-1], nums[mid], nums[mid+1])
    if nums[mid-1]==_max:
        swap(nums, mid, mid-1)
    elif nums[mid+1]==_max:
        swap(nums, mid, mid+1)
        
def swap(nums, i, j):
    nums[i], nums[j] = nums[j], nums[i]

nums = [5,3,1,2,3]
nums = [0,1,4,7,8,9]
print(peaksAndValleys(nums.copy()))
print(peaksAndValleys(nums.copy()))



        
    
    
    

