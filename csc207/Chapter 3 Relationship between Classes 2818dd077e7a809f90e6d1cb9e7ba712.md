# Chapter 3: Relationship between Classes

### 3.1 Inheritance

```java
class Child extends Parent {
...
}
```

- Child inherits all methods and variables defined in Parent

### 3.1.1 Abstract classes

- not meant to be initalized directly!!!!!!!
- any non-abstract child class have to implement the empty methods in the abstract class 😂

```java
abstract class AbstractClass{
    abstract void something();    // Abstract methods have no body!
}

class NonAbstract extends AbstractClass{
    void something(){
        ...   // Method body here!
    }
}
```

### 3.1.2 Overriding methods

- can override a parent class method by redefining it, with @Override annotation, informing compiler.
- Annotation is not required but helps prevent errors.

```java
class Parent{
		void something(){
		...
		}
}

class child extends Parent {
		@Override
		void something(){
		//something new
		}
}
```

### 3.2 Interfaces

- each class can only extend one single class (only one parent class)
- to describe more behaviours: interface defines a contract what a class can do
- In an interface:
    1. All methods are public and abstract (implicitly, no need include keywords ourselves!)
    2. All variables are public, static and final (implicitly) (no instance var!)
    3. Classes that implement this interface must provide implementation for the interface’s abstract methods
    4. After Java 8, interface can now include
        1. default methods: methods with a body
            - default void func() {}
        2. static methods: useful for utility behavior
    5. class can implement multiple interfaces by using commas after the ‘implements’ keyword.
    6. interfaces can extend other interfaces!

```java
class Corn extends Plant implements Edible, Washable {
    void eat() {
        ...    // Our implementation here!
    }
    
    // Overriding the default wash method
    @Override
    public void wash() {
        System.out.println("Thoroughly washing the corn...");
    }
}
```

### 3.3 super

- use `super()` to refer to methods in parent class
- `super.method()` to call parent’s method, for instance.
- `super()` or `super(a, b, c)` if need pass in parameters
- `super()` constructor call is implicit!

```java
class Child extends Parent {
    int attribute1;
    int attribute2;

    public Child(int a, int b) {
        super();  // Will auto-call this implicitly! 
        // However if the super() constructor takes no argument. 
        // If super() requires argument, then need manual write that in.
        this.attribute2 = b;
    }
}
```

### 3.4 Polymorphism

- polymorphism: ability of object to take on many forms
- the object can be treated as instance of its own class, all superclasses, any interface it implements…
- `class Dog extends Canine implements Domesticatable { ... }`
    - `Dog d = new Dog();`
    - Dog `(d instance of Dog)`
    - Canine`(d instance of Canine)`
    - Object`(d instance of Object)`
    - Domesticatable`(d instance of Domesticatable)`
    

### 3.5 Casting

- sometimes we want to convert between types, a general superclass or subclass.
- Casting OBJECTS: we are ‘re-labelling’ the reference type. Actual data remains same.
- Casting PRIMITIVES: converting value itself, may lead to loss of precision, and is IRREVERSIBLE! (implicit: int to double; explicit: double to int)
- **Upcasting:** assign subclass to superclass. Always safe!
- **Downcasting:** Only safe if actual object is an instance of the subclass!

```java
Animal a = new Dog(); // a dog is an animal so we can make this assignment
((Dog) a).bark(); // casting to Dog to access bark()
```

- Primitive Conversions

```java
int x = 1;
double y = 1.1;
double double_x = (double) x;
int int_y = (int) y;
```

### 3.6 Comparable

- Being comparable enables sorting. E.g. monthDate (?) sort by key?
- the .sort method requires all elements in the array must implement the `Comparable` interface!
- `class ItemClass implements Comparable<ItemClass> {}`

```java
int compareTo(Type obj)
    Compares this object with the specified object for order.
    Returns a negative integer, zero, or a positive integer
    as this object is less than, equal to, or greater than
    the specified object.
```

### 3.7 Comparator

- Out of syllabus (?)