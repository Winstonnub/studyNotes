# Chapter 1: Introduction to Java

# Chapter 1 Introduction to Java

### 1.0 Why learn java

- Java sits between Python and C in terms of abstraction. High level, static typing, manual compilation.

### 1.0.1 Running a Program

- Java uses **HYBRID of interpretation and compilation.**
- Compiler: javac translates source code into bytecode (intermediate form)
    - Java Virtual Machine JVM interprets and optimizes bytecode at runtime.
    - Terminal: javac [HelloWorld.java](http://HelloWorld.java) → generates HelloWorld.class
    - Terminal: java HelloWorld (to run HelloWorld.class bytecode)

### 1.0.2 Computer Architecture

- Machine code is not portable!
- Virtual Machine: provides consistent environment for running programs regardless of OS, achieve portability.
- High level to low level layers: Applications → VM (run .class bytecode) → OS → Hardware

### 1.1 A first look at Java

- no class can exist outside of a class
- basic structure:

```java
class Hello {
    public static void main(String[] args) {
      System.out.println(7 + 5);
    }
}
```

### 1.2 Variables

- type of variable can never change and has to be assigned.

```java
int i = 42
```

- Some error examples:
    - Type mismatch
    - cannot be resolved to a variable
    - duplicate local variable ( no need declare the type again! )

### 1.3 Reference Types and Primitive Types

- Primitive: begin with lowercases; directly store the value
- Reference: hold the reference to the object of Type

```java
public class Simple {
    public static void main(String[] args) {
        int age = 21;
        String name = "Jude";
        System.out.println("Ciao!");
    }
}
```

![image.png](Chapter%201%20Introduction%20to%20Java%202818dd077e7a80dbb95af66c242390ad/image.png)

- call stack: keep track of method that is running
- object space: where objects are stored
- static space: static members of classes are stored
- (stack frame): a container for the main method of class Simple

### 1.4 Strings

- need use double quotes!

```java
String s1 = new String("hello");
// SAME AS
String s1 = "hello" // Only works for string! Shortcut!
```

- StringPool
    - Initiate the string → is put into stringPool → if initiate another variable → search if the string is in StringPool → if Yes, then point it to that string object

```java
>>> s1 = "Hello"
>>> s2 = "Hello"
>>> s1 == s2
True

>>> String s1 = new String("Hello");
>>> String s2 = new String("Hello");
>>> s1 == s2; // NOT equal! But object instance eqaulity
False // we have two instance of String class
```

- Immutable: changing string returns a new one
- String Operations and Methods:
    - String s3 = s1 + s2 (creates a new string :))
    - .charAt(2) (means s1[2])
    - .substring(2,4) (means s1[2:4])
    - .trim (means s1.strip and remove all whitespaces)
    - length, startsWith, indexOf
- StringBuilder sb = new StringBuilder(”ban”) → MUTABLE
    - .append, .insert, .setCharAt, .reverse
- char: single character strings
- Mutation is faster than creating a new object.

### 1.5 Classes in Java

- use new to create object

### 1.6 Arrays

- int[] numbers = new int[5] ; // declare array of integers, {0,0,0,0,0}
- arrays are reference types
- length of array: .length
- access element: e.g. numbers[1] = 512;
    - do not offer slicing and negative indices

```java
Object[] miscellany = new Object[5];
miscellany[0] = new String("Songbird");

Object element = miscellany[0]; // WORKS!
String s = miscellany[0] // DOES NOT WORK! Unsure if is string!

String s = (String) miscellany[0] // cast it as String!
```

```java
// Irregular multidimensional arrays!
int[][] irregular;
irregular = new int[3][];
irregular[0] = new int[6];
irregular[1] = new int[99];
irregular[2] = new int[10];
irregular[1][8] = 170;
```

### 1.7 Aliases

- Cannot create aliases with primitives. (we just directly store!)
- Can create aliases with references.
    - String name = new String(”hi”); String s1 = name;
    - Both name and s1 points to one object “hi”.
- When we mutate the object, both aliases are affected.
- We can make another object of copy to avoid side effect of aliasing.
- Shallow Copy vs Deep Copy: TBD

### 1.8 Control Structures

- if statements: if (expression) {} else if {} else {}
- for loops, while loops:

```java
for (initialization; termination condition; increment){

}

// Enhanced for loop for collections:
for (int p : powers) {
  // for each integer p in powers
}

while (condition){

}

do {
} while (condition) // ensure body runs at least once
```

pre-increment: ++i, the value is increased before used

post-increment: i++, the value is increased after use

### 1.9 Parameters

- Skipped