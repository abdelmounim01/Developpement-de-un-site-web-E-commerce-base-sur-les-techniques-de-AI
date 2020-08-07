import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { NavbarvisiteurComponent } from './navbar.component';

describe('NavbarComponent', () => {
  let component: NavbarvisiteurComponent;
  let fixture: ComponentFixture<NavbarvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ NavbarvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(NavbarvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
