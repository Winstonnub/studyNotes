# Chapter 5: Java Gotchas and Subtleties

### 5.1 Shadowing

def f() → None:

```java
public class ShadowExample {
    private int shadowedVariable = 10;

    public void shadowingMethod(){
        int shadowedVariable = 20;
        System.out.println(shadowedVariable);
        System.out.println(this.shadowedVariable);
        // we use 'this' when we have dupes to refer to instance var!
    }
}

20
10

```

### 5.2 Array Copy

```java
int[] lst = {1, 2, 3};
int[] lstCopy = lst.clone(); // lstCopy = lst.copy()
int[] lstAlias = lst;

// lst and lstAlias refers to same list.
// lstCopy is on its own.

// using clone() create copy of outermost arrays, but not 
// copies of inner arrays. We need clone() all inner arrays to make a deeper copy.
```

### 5.3 Autoboxing

- Autoboxing: primitive to wrapper
- Unboxing : wrapper to primitive

```java
// Autoboxing
public class Example {
    public static void main(String[] args) {
        int primitiveInt = 10;

        // Autoboxing: primitive int → Integer object
        Integer wrapperInt = primitiveInt;

        System.out.println(wrapperInt); // prints 10
    }
}

// Unboxing
public class Example {
    public static void main(String[] args) {
        Integer wrapperInt = 20;

        // Unboxing: Integer object → primitive int
        int primitiveInt = wrapperInt;

        System.out.println(primitiveInt); // prints 20
    }
}

```

- Value Comparison

```java
Integer a = 100;
int b = 100;

System.out.println(a == b); // true — compares values; a is unboxed
```

- Reference comparison

```java
Integer x = new Integer(100);
Integer y = new Integer(100);

System.out.println(x == y); // false — different objects
System.out.println(x.equals(y)); // true — same value
```

- Integer caching (java caches integer values from -128 to 127)

```java
Integer a = 100;
Integer b = 100;
System.out.println(a == b); // true — same cached object

Integer c = 200;
Integer d = 200;
System.out.println(c == d); // false — not cached
```

- Null Safety

```java
Integer a = null;
int b = 100;

System.out.println(a == b); // throws NullPointerException
```